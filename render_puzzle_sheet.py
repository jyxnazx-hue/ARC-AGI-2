import json
import sys
from pathlib import Path
# pyrefly: ignore [missing-import]
from PIL import Image, ImageDraw,  ImageFont
# pyrefly: ignore [missing-import]
import numpy as np

from src.core.grid import Grid, ARC_ANSI_COLORS

PALETTE = ARC_ANSI_COLORS

def render_single_grid(cells: np.ndarray, cell_size=20, border_w=1, border_color=(255, 255, 255)) -> Image.Image:
    """Renders a grid with white grid cell dividers."""
    h, w = cells.shape
    img_w = w * cell_size + (w + 1) * border_w
    img_h = h * cell_size + (h + 1) * border_w
    img = Image.new("RGB", (img_w, img_h), color=border_color)
    draw = ImageDraw.Draw(img)

    for r in range(h):
        for c in range(w):
            fill = PALETTE.get(int(cells[r, c]), (0, 0, 0))
            x0 = border_w + c * (cell_size + border_w)
            y0 = border_w + r * (cell_size + border_w)
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], fill=fill)
    return img

def render_full_puzzle(task_path: str, output_path: str = None, cell_size: int = 20) -> Path:
    """
    Renders all train pairs and test pairs onto a single unified canvas.
    Layout matches standard ARC benchmark sheets:
    Train 1: [In] -> [Out] | Train 2: [In] -> [Out] ... | Test: [In] -> [?]
    """
    task_file = Path(task_path)
    with open(task_file, "r", encoding="utf-8") as f:
        task = json.load(f)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    columns = []  # Stores rendered column PIL images
    header_font = ImageFont.load_default()

    # 1. Build Demonstration Columns
    for idx, pair in enumerate(train_pairs):
        in_img = render_single_grid(np.array(pair["input"]), cell_size=cell_size)
        out_img = render_single_grid(np.array(pair["output"]), cell_size=cell_size)

        col_w = max(in_img.width, out_img.width)
        # 25px label + in_img + 20px gap/arrow + out_img + 10px margin
        col_h = 25 + in_img.height + 25 + out_img.height + 10
        col_canvas = Image.new("RGB", (col_w, col_h), color=(240, 242, 245))
        draw = ImageDraw.Draw(col_canvas)

        # Labels
        draw.text((4, 4), f"Train {idx + 1} Input ({np.array(pair['input']).shape[0]}x{np.array(pair['input']).shape[1]})", fill=(0, 0, 0), font=header_font)
        col_canvas.paste(in_img, ((col_w - in_img.width) // 2, 25))

        arrow_y = 25 + in_img.height + 4
        draw.text(((col_w // 2) - 5, arrow_y), "↓", fill=(50, 50, 50), font=header_font)

        out_y = arrow_y + 20
        draw.text((4, out_y - 16), f"Output ({np.array(pair['output']).shape[0]}x{np.array(pair['output']).shape[1]})", fill=(0, 0, 0), font=header_font)
        col_canvas.paste(out_img, ((col_w - out_img.width) // 2, out_y))
        columns.append(col_canvas)

    # 2. Build Test Challenge Column
    for idx, pair in enumerate(test_pairs):
        in_img = render_single_grid(np.array(pair["input"]), cell_size=cell_size)
        
        # Test output or question mark placeholder
        if "output" in pair:
            out_img = render_single_grid(np.array(pair["output"]), cell_size=cell_size)
            out_label = f"Expected ({np.array(pair['output']).shape[0]}x{np.array(pair['output']).shape[1]})"
        else:
            out_img = Image.new("RGB", in_img.size, color=(220, 220, 220))
            ImageDraw.Draw(out_img).text((out_img.width // 2 - 5, out_img.height // 2 - 5), "?", fill=(0, 0, 0), font=header_font)
            out_label = "Target Output (?)"

        col_w = max(in_img.width, out_img.width)
        col_h = 25 + in_img.height + 25 + out_img.height + 10
        col_canvas = Image.new("RGB", (col_w, col_h), color=(230, 235, 245))  # Distinct tint for Test
        draw = ImageDraw.Draw(col_canvas)

        draw.text((4, 4), f"TEST {idx + 1} Input ({np.array(pair['input']).shape[0]}x{np.array(pair['input']).shape[1]})", fill=(180, 0, 0), font=header_font)
        col_canvas.paste(in_img, ((col_w - in_img.width) // 2, 25))

        arrow_y = 25 + in_img.height + 4
        draw.text(((col_w // 2) - 5, arrow_y), "↓", fill=(180, 0, 0), font=header_font)

        out_y = arrow_y + 20
        draw.text((4, out_y - 16), out_label, fill=(180, 0, 0), font=header_font)
        col_canvas.paste(out_img, ((col_w - out_img.width) // 2, out_y))
        columns.append(col_canvas)

    # 3. Stitch Columns Horizontally with Separator Margins
    gap = 20
    title_height = 40
    total_w = sum(c.width for c in columns) + gap * (len(columns) + 1)
    max_h = max(c.height for c in columns) + title_height + 20

    sheet = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))
    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.text((20, 12), f"ARC-AGI-2 Task: {task_file.stem}", fill=(20, 20, 20), font=header_font)

    current_x = gap
    for c in columns:
        sheet.paste(c, (current_x, title_height))
        current_x += c.width + gap

    if not output_path:
        out_dir = Path("rendered_sheets")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{task_file.stem}.png"

    sheet.save(output_path)
    print(f"Generated unified sheet: {output_path}")
    return Path(output_path)

if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else str(next(Path("data/build").glob("*.json")))
    render_full_puzzle(task_arg)