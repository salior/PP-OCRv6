import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    from paddleocr import PaddleOCR
except ImportError as exc:
    raise SystemExit(
        "未安装 paddleocr。请先执行: pip install -r requirements.txt"
    ) from exc


NumberCandidate = Tuple[str, float]


@dataclass
class OCRConfig:
    lang: str = "en"
    device: str = "cpu"
    engine: str = "onnxruntime"
    min_score: float = 0.35


def build_ocr(config: OCRConfig) -> PaddleOCR:
    attempts = []
    pipeline_base = {
        "lang": config.lang,
        "device": config.device,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    legacy_base = {
        "lang": config.lang,
        "use_angle_cls": False,
    }

    preferred_engines = [config.engine] if config.engine != "auto" else ["onnxruntime", "paddle_dynamic", "paddle"]

    for engine in preferred_engines:
        if engine == "onnxruntime":
            attempts.append(
                {
                    **pipeline_base,
                    "ocr_version": "PP-OCRv6",
                    "engine": "onnxruntime",
                    "engine_config": {
                        "providers": ["CPUExecutionProvider"],
                    },
                }
            )
        elif engine == "paddle_dynamic":
            attempts.append(
                {
                    **pipeline_base,
                    "ocr_version": "PP-OCRv6",
                    "engine": "paddle_dynamic",
                }
            )
        elif engine == "paddle":
            attempts.append(
                {
                    **pipeline_base,
                    "ocr_version": "PP-OCRv6",
                }
            )

    attempts.extend(
        [
            {
                **pipeline_base,
            },
            legacy_base,
        ]
    )

    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("无法初始化 PaddleOCR")


def center_roi(frame: np.ndarray, roi_ratio: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    roi_width = max(80, int(width * roi_ratio))
    roi_height = max(60, int(height * roi_ratio * 0.35))
    left = max(0, (width - roi_width) // 2)
    top = max(0, (height - roi_height) // 2)
    right = min(width, left + roi_width)
    bottom = min(height, top + roi_height)
    return frame[top:bottom, left:right].copy(), (left, top, right, bottom)


def preprocess_variants(image: np.ndarray, scale: float) -> List[Tuple[str, np.ndarray]]:
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(thresh)

    # 自适应阈值：对 LCD 背光不均匀的情况更鲁棒，能更好保留小数点
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 5)
    adaptive_inv = cv2.bitwise_not(adaptive)

    # 形态学闭运算：连接断裂的笔画，同时保留小圆点
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return [
        ("raw", scaled),
        ("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)),
        ("thresh", cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)),
        ("inv", cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)),
        ("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)),
        ("adaptive_inv", cv2.cvtColor(adaptive_inv, cv2.COLOR_GRAY2BGR)),
        ("closed", cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)),
    ]


def normalize_number(text: str) -> Optional[str]:
    cleaned = text.replace("O", "0").replace("o", "0").replace(" ", "")
    cleaned = cleaned.replace("，", ",").replace("。", ".")
    # OCR 经常把小数点识别成这些字符
    cleaned = cleaned.replace("·", ".").replace("•", ".").replace("°", ".")
    cleaned = cleaned.replace("．", ".")  # 全角句点
    # 去除首尾非数字非小数点的干扰字符
    cleaned = re.sub(r"^[^\d+-]+|[^\d.]+$", "", cleaned)
    matches = re.findall(r"[-+]?\d+(?:[.,]\d+)?", cleaned)
    if not matches:
        return None
    candidate = max(matches, key=len)
    return candidate.replace(",", ".")


def detect_decimal_points(roi_gray: np.ndarray, scale: float) -> List[float]:
    """在灰度图中检测可能是小数点的小圆点，返回它们的相对 x 坐标 (0~1)。"""
    # 二值化
    _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 去除大块区域（数字笔画），只保留小连通域（小数点）
    # 先做开运算去掉细线
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    img_h, img_w = roi_gray.shape[:2]
    # 小数点的面积范围（相对于图像尺寸自适应）
    min_area = max(4, int(img_w * img_h * 0.0002))
    max_area = max(50, int(img_w * img_h * 0.008))
    # 小数点的长宽比应该接近 1:1
    min_aspect = 0.3

    dot_positions: List[float] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if aspect < min_aspect:
            continue
        # 小数点一般在数字的底部区域（下半 60%）
        if y > img_h * 0.3:
            dot_positions.append((x + w / 2) / img_w)

    return sorted(dot_positions)


def apply_decimal_heuristic(
    candidates: List[NumberCandidate],
    dot_positions: List[float],
) -> List[NumberCandidate]:
    """如果 OCR 结果没有小数点，但检测到了疑似小数点，尝试插入小数点。"""
    if not dot_positions:
        return candidates

    enhanced: List[NumberCandidate] = list(candidates)
    for value, score in candidates:
        if "." in value:
            continue  # 已经有小数点了
        digits = re.sub(r"[^\d]", "", value)
        if len(digits) < 2:
            continue
        sign = ""
        if value and value[0] in "+-":
            sign = value[0]
        # 对每个检测到的点位置，尝试在对应位置插入小数点
        for dot_x in dot_positions:
            insert_pos = round(dot_x * len(digits))
            insert_pos = max(1, min(insert_pos, len(digits) - 1))
            new_value = sign + digits[:insert_pos] + "." + digits[insert_pos:]
            # 置信度稍微降低一点，因为是重建的
            enhanced.append((new_value, score * 0.9))

    return enhanced


def extract_from_legacy(result: Sequence[Sequence[object]]) -> List[NumberCandidate]:
    candidates: List[NumberCandidate] = []
    for line in result:
        if len(line) < 2:
            continue
        text_info = line[1]
        if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
            continue
        text = str(text_info[0])
        score = float(text_info[1])
        value = normalize_number(text)
        if value is not None:
            candidates.append((value, score))
    return candidates


def extract_from_predict_result(item: object) -> List[NumberCandidate]:
    candidates: List[NumberCandidate] = []
    payload = item
    if hasattr(item, "to_dict"):
        payload = item.to_dict()
    elif hasattr(item, "res"):
        payload = getattr(item, "res")

    if isinstance(payload, dict):
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or payload.get("rec_score") or []
        if isinstance(scores, (int, float)):
            scores = [float(scores)] * len(texts)
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 0.0
            value = normalize_number(str(text))
            if value is not None:
                candidates.append((value, score))
    return candidates


def run_ocr(ocr: PaddleOCR, image: np.ndarray) -> List[NumberCandidate]:
    try:
        results = ocr.predict(image)
        candidates: List[NumberCandidate] = []
        for item in results:
            candidates.extend(extract_from_predict_result(item))
        if candidates:
            return candidates
    except Exception:
        pass

    candidates = []
    try:
        legacy_result = ocr.ocr(image, cls=False)
        for block in legacy_result:
            if isinstance(block, list):
                candidates.extend(extract_from_legacy(block))
    except Exception:
        return []
    return candidates


def pick_best_candidate(candidates: Iterable[NumberCandidate], min_score: float) -> Optional[NumberCandidate]:
    filtered = [item for item in candidates if item[1] >= min_score]
    if not filtered:
        return None
    return max(filtered, key=lambda item: (item[1], len(item[0])))


def draw_overlay(
    frame: np.ndarray,
    roi_box: Tuple[int, int, int, int],
    current_value: str,
    current_score: float,
    source_name: str,
    fps: float,
) -> np.ndarray:
    display = frame.copy()
    left, top, right, bottom = roi_box
    cv2.rectangle(display, (left, top), (right, bottom), (0, 200, 255), 2)

    panel_color = (18, 18, 18)
    cv2.rectangle(display, (12, 12), (620, 118), panel_color, -1)
    cv2.putText(display, f"Value: {current_value}", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 120), 2)
    cv2.putText(display, f"Score: {current_score:.3f}  Source: {source_name}", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"FPS: {fps:.1f}  Keys: q=quit  c=toggle ROI/full", (24, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 190, 190), 1)
    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用摄像头识别液晶数字的 PP-OCRv6 示例")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--width", type=int, default=1280, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="摄像头高度")
    parser.add_argument("--roi-ratio", type=float, default=0.55, help="中心识别框占画面宽度比例")
    parser.add_argument("--scale", type=float, default=2.0, help="OCR 前放大倍数")
    parser.add_argument("--interval", type=float, default=0.35, help="OCR 间隔秒数")
    parser.add_argument("--lang", default="en", help="OCR 语言，液晶数字建议 en")
    parser.add_argument("--device", default="cpu", help="cpu 或 gpu")
    parser.add_argument("--engine", default="onnxruntime", choices=["onnxruntime", "paddle_dynamic", "paddle", "auto"], help="推理引擎，Windows CPU 建议 onnxruntime")
    parser.add_argument("--min-score", type=float, default=0.35, help="最小置信度")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ocr = build_ocr(OCRConfig(lang=args.lang, device=args.device, engine=args.engine, min_score=args.min_score))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("无法打开摄像头，请确认摄像头编号是否正确。", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    current_value = "--"
    current_score = 0.0
    current_source = "none"
    use_roi_only = True
    last_ocr_at = 0.0
    last_frame_at = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取摄像头画面失败。", file=sys.stderr)
                break

            roi_image, roi_box = center_roi(frame, args.roi_ratio)
            target = roi_image if use_roi_only else frame

            now = time.perf_counter()
            elapsed = now - last_frame_at
            last_frame_at = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            if now - last_ocr_at >= args.interval:
                candidates: List[NumberCandidate] = []
                best_source = "none"

                # 检测小数点位置（在放大后的灰度图上做轮廓分析）
                scaled_for_dot = cv2.resize(target, None, fx=args.scale, fy=args.scale,
                                            interpolation=cv2.INTER_CUBIC)
                gray_for_dot = cv2.cvtColor(scaled_for_dot, cv2.COLOR_BGR2GRAY)
                dot_positions = detect_decimal_points(gray_for_dot, args.scale)

                for source_name, variant in preprocess_variants(target, args.scale):
                    variant_candidates = run_ocr(ocr, variant)
                    if variant_candidates:
                        # 对没有小数点的候选结果，尝试用检测到的点重建
                        variant_candidates = apply_decimal_heuristic(variant_candidates, dot_positions)
                        candidates.extend(variant_candidates)
                        best_source = source_name
                best = pick_best_candidate(candidates, args.min_score)
                if best is not None:
                    current_value, current_score = best
                    current_source = best_source
                last_ocr_at = now

            screen = draw_overlay(frame, roi_box, current_value, current_score, current_source, fps)
            cv2.imshow("PP-OCRv6 LCD Camera Demo", screen)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                use_roi_only = not use_roi_only
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())