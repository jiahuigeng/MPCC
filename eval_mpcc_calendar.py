import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd


# -----------------------------
# Utilities: normalization + exact match
# -----------------------------
def extract_time_slot(text: str) -> Optional[Dict[str, str]]:
    """
    Extracts start_time and end_time from a JSON string (or partial JSON).
    Returns dict with keys 'start_time', 'end_time' or None if failed.
    """
    if not text:
        return None
        
    # Try to find JSON block
    import re
    import json
    
    # 1. Try to parse as pure JSON
    try:
        data = json.loads(text)
        # Check if it's the gold format which might be {'best': '[{...}]'}
        if isinstance(data, dict):
            if "best" in data:
                # Recursively parse the string in 'best'
                try:
                    inner = json.loads(data["best"])
                    if isinstance(inner, list) and len(inner) > 0:
                        data = inner[0]
                except:
                    pass
            
            if "start_time" in data and "end_time" in data:
                return {"start_time": str(data["start_time"]).strip(), "end_time": str(data["end_time"]).strip()}
        
        # Check if it's a list (some gold answers might be list)
        if isinstance(data, list) and len(data) > 0:
             if isinstance(data[0], dict) and "start_time" in data[0]:
                 return {"start_time": str(data[0]["start_time"]).strip(), "end_time": str(data[0]["end_time"]).strip()}

    except:
        pass

    # 2. Try to find markdown json block
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "start_time" in data and "end_time" in data:
                return {"start_time": str(data["start_time"]).strip(), "end_time": str(data["end_time"]).strip()}
        except:
            pass

    # 3. Fallback: regex search for keys directly in text
    # Looking for "start_time": "XX:XX PM"
    try:
        start_match = re.search(r'"start_time"\s*:\s*"([^"]+)"', text)
        end_match = re.search(r'"end_time"\s*:\s*"([^"]+)"', text)
        if start_match and end_match:
             return {"start_time": start_match.group(1).strip(), "end_time": end_match.group(1).strip()}
    except:
        pass
        
    return None

def normalize_text(s: str) -> str:
    """Simple normalization for exact match."""
    if not s:
        return ""
    # Strip markdown code blocks
    s = s.strip()
    if s.startswith("```"):
        # Remove first line (```json or similar)
        lines = s.split('\n')
        if len(lines) > 1:
            # If starts with ``` remove first line
            if lines[0].startswith("```"):
                lines = lines[1:]
            # If ends with ``` remove last line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines)
        else:
            # Single line case like ```json...```
            s = s.replace("```json", "").replace("```", "")
            
    return " ".join(s.split()).lower()


def parse_gold_answer(gold: Any) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Parses the Gold Answer to extract 'best' options and 'feasible' options.
    Returns: (best_list, feasible_list)
    Each list contains dicts with {'start_time', 'end_time'}.
    """
    import json
    import ast
    
    best_list = []
    feasible_list = []
    
    # Debug raw gold
    # print(f"DEBUG: Raw gold type: {type(gold)}")
    # print(f"DEBUG: Raw gold val: {gold}")

    # Ensure gold is a dict
    data = gold
    if not isinstance(data, dict):
        # 1. Try ast.literal_eval (handles python dict string like {'a': 1})
        try:
            data = ast.literal_eval(str(gold))
        except:
            # 2. Try json.loads (handles json string like {"a": 1})
            try:
                data = json.loads(str(gold))
            except:
                # 3. Try simple replace fix
                try:
                     data = json.loads(str(gold).replace("'", '"'))
                except:
                     pass

    if not isinstance(data, dict):
        return [], []

    def normalize_slot(slot):
        if isinstance(slot, dict) and "start_time" in slot and "end_time" in slot:
            return {
                "start_time": str(slot["start_time"]).strip(),
                "end_time": str(slot["end_time"]).strip()
            }
        return None

    def parse_inner_list(val):
        """Parses the value of 'best' or 'feasible', which could be a list or a stringified list."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            # Try json load first
            try:
                return json.loads(val)
            except:
                pass
            # Try ast literal eval
            try:
                return ast.literal_eval(val)
            except:
                pass
        return []

    # Parse 'best'
    if "best" in data:
        raw_best = data["best"]
        parsed_best = parse_inner_list(raw_best)
        if isinstance(parsed_best, list):
            for item in parsed_best:
                slot = normalize_slot(item)
                if slot:
                    best_list.append(slot)
        elif isinstance(parsed_best, dict):
             # sometimes best is a single dict?
             slot = normalize_slot(parsed_best)
             if slot:
                 best_list.append(slot)

    # Parse 'feasible'
    if "feasible" in data:
        raw_feas = data["feasible"]
        parsed_feas = parse_inner_list(raw_feas)
        if isinstance(parsed_feas, list):
            for item in parsed_feas:
                slot = normalize_slot(item)
                if slot:
                    feasible_list.append(slot)
    
    return best_list, feasible_list

def check_match(pred_slot: Dict[str, str], gold_list: List[Dict[str, str]]) -> bool:
    """Checks if pred_slot matches ANY item in gold_list."""
    if not pred_slot:
        return False
    
    for gold in gold_list:
        if (gold["start_time"].lower() == pred_slot["start_time"].lower() and 
            gold["end_time"].lower() == pred_slot["end_time"].lower()):
            return True
    return False

def exact_match(pred: str, gold: Any) -> Tuple[bool, bool]:
    """
    Returns (is_optimal, is_feasible).
    is_optimal: Matches any option in 'best'
    is_feasible: Matches any option in 'feasible' (note: best is usually a subset of feasible)
    """
    pred_slot = extract_time_slot(pred)
    best_list, feasible_list = parse_gold_answer(gold)
    
    is_optimal = check_match(pred_slot, best_list)
    
    # If feasible list is empty but best is not, assume best are the only feasible ones?
    # Usually feasible list should cover everything. 
    # If pred matches best, it is automatically feasible (usually).
    # But let's check strictly against feasible_list if it exists.
    
    is_feasible = False
    if feasible_list:
        is_feasible = check_match(pred_slot, feasible_list)
    else:
        # Fallback: if no feasible list provided, maybe optimal implies feasible
        is_feasible = is_optimal

    return is_optimal, is_feasible


# -----------------------------
# Model interface (plug-in)
# -----------------------------
class BaseVLM:
    """
    Implement `generate(question, image_path, image_bytes, meta)` -> str
    """
    def generate(
        self,
        question: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> str:
        raise NotImplementedError


class DummyEchoModel(BaseVLM):
    """
    Baseline to sanity-check the pipeline.
    It returns empty string by default (0% expected).
    """
    def generate(self, question, image_path=None, image_bytes=None, meta=None) -> str:
        return ""


import base64
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Import API Key
try:
    from api_sources import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class OpenAIVisionModel(BaseVLM):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        if OpenAI is None:
            raise ImportError("Please install openai: pip install openai")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in api_sources.py or environment variables")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.prompt_prefix = (
            "You are solving MPCC Calendar Planning. "
            "Read the calendar image carefully and answer EXACTLY in the required output format.\n\n"
        )

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(self, question, image_path=None, image_bytes=None, meta=None) -> str:
        if image_path is None and image_bytes is None:
            raise ValueError("Need image_path or image_bytes for vision model.")
        
        content_payload = [{"type": "text", "text": self.prompt_prefix + f"Question: {question}\n"}]

        # Add images
        if image_path is not None:
             # If it's a list of paths
            if isinstance(image_path, list):
                for p in image_path:
                    b64 = self._encode_image(p)
                    content_payload.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    )
            else:
                b64 = self._encode_image(image_path)
                content_payload.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )
        elif image_bytes is not None:
            # If it's a list of bytes
            if isinstance(image_bytes, list):
                 print(f"    DEBUG: Sending {len(image_bytes)} images to API.")
                 for i, b in enumerate(image_bytes):
                      b64 = base64.b64encode(b).decode("utf-8")
                      content_payload.append(
                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                     )
                      print(f"    DEBUG: Added image {i+1}/{len(image_bytes)} to payload.")
            else:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                content_payload.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": content_payload
                }],
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
    needed = {"question", "answer", "difficulty"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")
    return df


def resolve_image(example: Dict[str, Any], base_dir: str) -> Tuple[Optional[str], Optional[List[bytes]]]:
    """
    Prefer image_path. If missing, try image bytes in 'image'.
    Returns a LIST of image bytes (since MPCC examples can have multiple images).
    """
    # ... (skipping path logic for now as we know files aren't there) ...
    # Fallback: image bytes in 'image'
    img = example.get("image")
    if img is None:
        return None, None

    images_bytes_list = []

    # Handle string-encoded images
    if isinstance(img, str):
        try:
            import ast
            # Try to parse string as list
            if img.strip().startswith('[') and img.strip().endswith(']'):
                img_list = ast.literal_eval(img)
                # Now img_list is a list of base64 strings
                for img_str in img_list:
                    if isinstance(img_str, str):
                        if img_str.startswith('data:image'):
                            img_str = img_str.split(',')[1]
                        try:
                            images_bytes_list.append(base64.b64decode(img_str))
                        except:
                            pass
            else:
                 # Single string
                 if img.startswith('data:image'):
                     img = img.split(',')[1]
                 try:
                     images_bytes_list.append(base64.b64decode(img))
                 except:
                     pass
        except:
            pass

    # If it's already a list of something else (though in parquet it's usually stringified list)
    elif isinstance(img, list):
         for item in img:
             if isinstance(item, str):
                 if item.startswith('data:image'):
                     item = item.split(',')[1]
                 try:
                     images_bytes_list.append(base64.b64decode(item))
                 except:
                     pass
             elif isinstance(item, (bytes, bytearray)):
                 images_bytes_list.append(bytes(item))

    if images_bytes_list:
        return None, images_bytes_list
    
    return None, None


# -----------------------------
# Metrics & evaluation
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


def evaluate_calendar(
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

    for i, row in df.iterrows():
        ex = row.to_dict()
        q = ex["question"]
        gold = ex["answer"]

        image_path, image_bytes = resolve_image(ex, base_dir)
        
        # Debug / Info logging for image loading
        img_status = "❌ Missing"
        if image_path:
            count = len(image_path) if isinstance(image_path, list) else 1
            img_status = f"✅ File(s): {count}"
        elif image_bytes:
            count = len(image_bytes) if isinstance(image_bytes, list) else 1
            img_status = f"✅ Byte(s): {count} image(s)"
        
        print(f"[{i+1}/{len(df)}] ID: {ex.get('index')} | Image: {img_status}")
        print(f"    Question: {q}")
        print(f"    Image Path (Metadata): {ex.get('image_path')}")

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

        # DEBUG: Print extraction results
        # gold_slot = extract_time_slot(str(gold)) if not isinstance(gold, dict) else extract_time_slot(json.dumps(gold))
        pred_slot = extract_time_slot(pred)
        
        best_list, feasible_list = parse_gold_answer(gold)
        print(f"    [Parsed] Best Options: {best_list}")
        print(f"    [Parsed] Feasible Options (Total {len(feasible_list)})")
        # Only print first few feasible options to avoid clutter
        if len(feasible_list) > 3:
            print(f"             {feasible_list[:3]} ...")
        else:
            print(f"             {feasible_list}")
            
        print(f"    [Extracted] Pred: {pred_slot}")

        is_optimal, is_feasible = exact_match(pred, gold)
        
        if is_optimal:
            optimal_hits += 1
        if is_feasible:
            feasible_hits += 1

        status_str = []
        if is_optimal: status_str.append("🌟 OPTIMAL")
        if is_feasible: status_str.append("✅ FEASIBLE")
        if not status_str: status_str.append("❌ FAIL")
        
        print(f"    Result: {' | '.join(status_str)}")
        if not is_optimal:
             # print(f"    Gold (Best): {parse_gold_answer(gold)[0]}")
             # print(f"    Pred: {str(pred)}")
             pass
        print("-" * 50)
        
        rows_out.append({
            "index": ex.get("index"),
            "question": q,
            "gold": gold,
            "prediction": pred,
            "is_optimal": is_optimal,
            "is_feasible": is_feasible
        })

    # Summary
    print(f"=== MPCC Calendar Planning Evaluation ===")
    print(f"File: {parquet_path}")
    print(f"Total: {len(df)}")
    print(f"Feasible Plan Rate: {feasible_hits/len(df):.4f} ({feasible_hits}/{len(df)})")
    print(f"Optimal Plan Rate:  {optimal_hits/len(df):.4f} ({optimal_hits}/{len(df)})")
    
    if save_predictions:
        print(f"Saving predictions to {save_predictions}...")
        with open(save_predictions, "w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
                
    return EvalResult(
        total=len(df),
        feasible_hits=feasible_hits,
        optimal_hits=optimal_hits
    )


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True, help="Path to calendar_plan_*.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N examples")
    parser.add_argument("--save", type=str, default=None, help="Save predictions to a JSONL file")
    parser.add_argument("--model", type=str, default="dummy", choices=["dummy", "openai"], help="Model backend (extendable)")
    args = parser.parse_args()

    if args.model == "dummy":
        model = DummyEchoModel()
    elif args.model == "openai":
        model = OpenAIVisionModel(model_name="gpt-4o")
    else:
        raise ValueError("Unknown model")

    res = evaluate_calendar(
        parquet_path=args.parquet,
        model=model,
        limit=args.limit,
        save_predictions=args.save
    )

    print("=== MPCC Calendar Planning Evaluation ===")
    print(f"File: {args.parquet}")
    print(f"Total: {res.total}")
    print(f"Feasible Plan Rate: {res.feasible_rate:.4f} ({res.feasible_hits}/{res.total})")
    print(f"Optimal Plan Rate:  {res.optimal_rate:.4f} ({res.optimal_hits}/{res.total})")


if __name__ == "__main__":
    main()
