import os
import re
import json
import argparse
import ast
import base64
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
TASK_CONFIG = {
    "meeting": {
        "keys": ["meeting_place", "start_time", "end_time"],
        "prompt": "You are solving MPCC Meeting Planning. Read the schedule and cost information carefully and answer EXACTLY in the required output format.\n\n"
    }
}

# -----------------------------
# Utilities: normalization + exact match
# -----------------------------
def normalize_val(v: Any) -> str:
    if isinstance(v, (int, float)):
        return str(v).strip().lower()
    return str(v).strip().lower()

def extract_answer_json(text: str, keys: List[str]) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object containing the specified keys.
    """
    if not text:
        return None
    
    # Helper to check if a dict has all keys
    def has_keys(d):
        return all(k in d for k in keys)

    # 1. Try to parse as pure JSON
    try:
        data = json.loads(text)
        # Check for gold format wrapper
        if isinstance(data, dict):
            if "best" in data:
                try:
                    inner = json.loads(data["best"])
                    if isinstance(inner, list) and len(inner) > 0:
                        data = inner[0]
                except:
                    pass
            if has_keys(data):
                return {k: data[k] for k in keys}
        
        if isinstance(data, list) and len(data) > 0:
             if isinstance(data[0], dict) and has_keys(data[0]):
                 return {k: data[0][k] for k in keys}
    except:
        pass

    # 2. Try to find markdown json block
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if has_keys(data):
                return {k: data[k] for k in keys}
        except:
            pass

    # 3. Fallback: regex search for keys directly in text
    extracted = {}
    try:
        for k in keys:
            # Match "key": "value" or "key": value (number)
            # This regex handles string values in quotes or number values
            m = re.search(rf'"{k}"\s*:\s*("([^"]+)"|([0-9.]+))', text)
            if m:
                if m.group(2): # Quoted string
                    extracted[k] = m.group(2)
                elif m.group(3): # Number
                    extracted[k] = m.group(3)
        
        if has_keys(extracted):
            return extracted
    except:
        pass
        
    return None

def parse_gold_answer(gold: Any, keys: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses the Gold Answer to extract 'best' options and 'feasible' options.
    """
    best_list = []
    feasible_list = []
    
    data = gold
    if not isinstance(data, dict):
        try:
            data = ast.literal_eval(str(gold))
        except:
            try:
                data = json.loads(str(gold))
            except:
                try:
                     data = json.loads(str(gold).replace("'", '"'))
                except:
                     pass

    if not isinstance(data, dict):
        return [], []

    def normalize_slot(slot):
        if isinstance(slot, dict) and all(k in slot for k in keys):
            return {k: slot[k] for k in keys}
        return None

    def parse_inner_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            try:
                return json.loads(val)
            except:
                pass
            try:
                return ast.literal_eval(val)
            except:
                pass
        return []

    # Parse 'best'
    if "best" in data:
        parsed_best = parse_inner_list(data["best"])
        if isinstance(parsed_best, list):
            for item in parsed_best:
                slot = normalize_slot(item)
                if slot:
                    best_list.append(slot)
        elif isinstance(parsed_best, dict):
             slot = normalize_slot(parsed_best)
             if slot:
                 best_list.append(slot)

    # Parse 'feasible'
    if "feasible" in data:
        parsed_feas = parse_inner_list(data["feasible"])
        if isinstance(parsed_feas, list):
            for item in parsed_feas:
                slot = normalize_slot(item)
                if slot:
                    feasible_list.append(slot)
    
    return best_list, feasible_list

def check_match(pred_slot: Dict[str, Any], gold_list: List[Dict[str, Any]], keys: List[str]) -> bool:
    """Checks if pred_slot matches ANY item in gold_list."""
    if not pred_slot:
        return False
    
    for gold in gold_list:
        match = True
        for k in keys:
            # Handle numeric comparison loosely
            v_gold = normalize_val(gold.get(k))
            v_pred = normalize_val(pred_slot.get(k))
            
            # Simple float comparison if both look like floats
            try:
                f_gold = float(v_gold)
                f_pred = float(v_pred)
                if abs(f_gold - f_pred) > 1e-4: # Tolerance
                    match = False
                    break
            except:
                # String comparison
                if v_gold != v_pred:
                    match = False
                    break
        if match:
            return True
    return False

def exact_match(pred: str, gold: Any, keys: List[str]) -> Tuple[bool, bool, Dict[str, Any]]:
    pred_slot = extract_answer_json(pred, keys)
    best_list, feasible_list = parse_gold_answer(gold, keys)
    
    is_optimal = check_match(pred_slot, best_list, keys)
    
    is_feasible = False
    if feasible_list:
        is_feasible = check_match(pred_slot, feasible_list, keys)
    else:
        is_feasible = is_optimal

    return is_optimal, is_feasible, pred_slot

# -----------------------------
# Model interface
# -----------------------------
class BaseVLM:
    def generate(self, question, image_path=None, image_bytes=None, meta=None) -> str:
        raise NotImplementedError

class DummyEchoModel(BaseVLM):
    def generate(self, question, image_path=None, image_bytes=None, meta=None) -> str:
        return ""

import base64
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from api_sources import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class OpenAIVisionModel(BaseVLM):
    def __init__(self, model_name: str = "gpt-4o-mini", prompt_prefix: str = ""):
        if OpenAI is None:
            raise ImportError("Please install openai: pip install openai")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in api_sources.py or environment variables")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.prompt_prefix = prompt_prefix

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(self, question, image_path=None, image_bytes=None, meta=None) -> str:
        if image_path is None and image_bytes is None:
            raise ValueError("Need image_path or image_bytes for vision model.")
        
        content_payload = [{"type": "text", "text": self.prompt_prefix + f"Question: {question}\n"}]

        # Add images
        if image_path is not None:
            if isinstance(image_path, list):
                for p in image_path:
                    b64 = self._encode_image(p)
                    content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            else:
                b64 = self._encode_image(image_path)
                content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        elif image_bytes is not None:
            if isinstance(image_bytes, list):
                 print(f"    DEBUG: Sending {len(image_bytes)} images to API.")
                 for i, b in enumerate(image_bytes):
                      b64 = base64.b64encode(b).decode("utf-8")
                      content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                      print(f"    DEBUG: Added image {i+1}/{len(image_bytes)} to payload.")
            else:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content_payload}],
                temperature=0.0,
                max_tokens=4096
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

# -----------------------------
# Data loading helpers
# -----------------------------
def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df

def resolve_image(example: Dict[str, Any], base_dir: str) -> Tuple[Optional[str], Optional[List[bytes]]]:
    img = example.get("image")
    if img is None:
        return None, None
    images_bytes_list = []
    if isinstance(img, str):
        try:
            if img.strip().startswith('[') and img.strip().endswith(']'):
                img_list = ast.literal_eval(img)
                for img_str in img_list:
                    if isinstance(img_str, str):
                        if img_str.startswith('data:image'): img_str = img_str.split(',')[1]
                        try: images_bytes_list.append(base64.b64decode(img_str))
                        except: pass
            else:
                 if img.startswith('data:image'): img = img.split(',')[1]
                 try: images_bytes_list.append(base64.b64decode(img))
                 except: pass
        except: pass
    elif isinstance(img, list):
         for item in img:
             if isinstance(item, str):
                 if item.startswith('data:image'): item = item.split(',')[1]
                 try: images_bytes_list.append(base64.b64decode(item))
                 except: pass
             elif isinstance(item, (bytes, bytearray)):
                 images_bytes_list.append(bytes(item))
    if images_bytes_list:
        return None, images_bytes_list
    return None, None

# -----------------------------
# Main Evaluation
# -----------------------------
@dataclass
class EvalResult:
    total: int
    feasible_hits: int
    optimal_hits: int

    @property
    def feasible_rate(self) -> float:
        return self.feasible_hits / self.total if self.total else 0.0

    @property
    def optimal_rate(self) -> float:
        return self.optimal_hits / self.total if self.total else 0.0

def evaluate_meeting(
    parquet_path: str,
    model: BaseVLM,
    limit: Optional[int] = None,
    save_predictions: Optional[str] = None
) -> EvalResult:
    base_dir = os.path.dirname(parquet_path)
    df = load_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit)

    feasible_hits = 0
    optimal_hits = 0
    rows_out: List[Dict[str, Any]] = []

    config = TASK_CONFIG["meeting"]
    keys = config["keys"]
    print(f"Task: MEETING PLANNING")
    print(f"Extraction Keys: {keys}")

    # Set prompt prefix if model supports it
    if hasattr(model, "prompt_prefix"):
        model.prompt_prefix = config["prompt"]

    for i, row in df.iterrows():
        ex = row.to_dict()
        q = ex["question"]
        gold = ex["answer"]
        
        image_path, image_bytes = resolve_image(ex, base_dir)
        
        img_status = "❌ Missing"
        if image_path:
            count = len(image_path) if isinstance(image_path, list) else 1
            img_status = f"✅ File(s): {count}"
        elif image_bytes:
            count = len(image_bytes) if isinstance(image_bytes, list) else 1
            img_status = f"✅ Byte(s): {count} image(s)"
        
        print(f"[{i+1}/{len(df)}] ID: {ex.get('index')} | Image: {img_status}")
        
        try:
            pred = model.generate(
                question=q,
                image_path=image_path,
                image_bytes=image_bytes,
                meta={"difficulty": ex.get("difficulty"), "category": ex.get("category"), "index": ex.get("index")}
            )
        except Exception as e:
            print(f"    Error during generation: {e}")
            pred = ""

        best_list, feasible_list = parse_gold_answer(gold, keys)
        
        is_optimal, is_feasible, pred_slot = exact_match(pred, gold, keys)
        
        print(f"    [Parsed] Best Options: {best_list}")
        print(f"    [Parsed] Feasible Options (Total {len(feasible_list)})")
        if len(feasible_list) > 3:
            print(f"             {feasible_list[:3]} ...")
        else:
            print(f"             {feasible_list}")
            
        print(f"    [Extracted] Pred: {pred_slot}")
        
        if is_optimal: optimal_hits += 1
        if is_feasible: feasible_hits += 1

        status_str = []
        if is_optimal: status_str.append("🌟 OPTIMAL")
        if is_feasible: status_str.append("✅ FEASIBLE")
        if not status_str: status_str.append("❌ FAIL")
        print(f"    Result: {' | '.join(status_str)}")
        print("-" * 50)
        
        rows_out.append({
            "index": ex.get("index"),
            "image_path": str(ex.get("image_path")),
            "question": q,
            "answer": gold,
            "predicted": pred,
            "Extracted_Pred": pred_slot,
            "is_optimal": is_optimal,
            "is_feasible": is_feasible
        })

    print(f"=== MPCC Evaluation (Meeting) ===")
    print(f"File: {parquet_path}")
    print(f"Total: {len(df)}")
    print(f"Feasible Plan Rate: {feasible_hits/len(df):.4f} ({feasible_hits}/{len(df)})")
    print(f"Optimal Plan Rate:  {optimal_hits/len(df):.4f} ({optimal_hits}/{len(df)})")
    
    if save_predictions:
        csv_path = save_predictions
        if not csv_path.endswith(".csv"):
             csv_path = os.path.splitext(csv_path)[0] + ".csv"
             
        print(f"Saving CSV to {csv_path}...")
        df_out = pd.DataFrame(rows_out)
        cols = ["image_path", "question", "answer", "predicted", "Extracted_Pred", "is_optimal", "is_feasible"]
        final_cols = [c for c in cols if c in df_out.columns]
        df_out[final_cols].to_csv(csv_path, index=False)

    return EvalResult(
        total=len(df),
        feasible_hits=feasible_hits,
        optimal_hits=optimal_hits
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True, help="Path to meeting_plan_*.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N examples")
    parser.add_argument("--save", type=str, default=None, help="Path to save output")
    parser.add_argument("--model", type=str, default="dummy", choices=["dummy", "openai"], help="Model backend")
    args = parser.parse_args()

    if args.model == "dummy":
        model = DummyEchoModel()
    elif args.model == "openai":
        model = OpenAIVisionModel(model_name="gpt-4o-mini")
    else:
        raise ValueError("Unknown model")

    evaluate_meeting(
        parquet_path=args.parquet,
        model=model,
        limit=args.limit,
        save_predictions=args.save
    )

if __name__ == "__main__":
    main()
