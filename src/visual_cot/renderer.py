import json
from pathlib import Path
from typing import Union, Tuple
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image, ImageDraw, ImageFont

from src.core.grid import Grid, ARC_ANSI_COLORS

PALETTE = ARC_ANSI_COLORS

def get_font(size: int = 14) -> ImageFont.ImageFont:
    """Loads a clean TrueType font on Windows, falling back to PIL default."""
    for font_name in ["arial.ttf", "segoeui.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()

def render_grid_image(
    grid_source: Union[Grid, np.ndarray, list],
    cell_size: int = 28,
    border_width: int = 1,
    border_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Renders a single grid with crisp cell dividing lines."""
    if isinstance(grid_source, Grid):
        cells = grid_source.cells
    elif isinstance(grid_source, np.ndarray):
        cells = grid_source
    else:
        cells = np.array(grid_source, dtype=np.int8)

    height, width = cells.shape
    img_width = width * cell_size + (width + 1) * border_width
    img_height = height * cell_size + (height + 1) * border_width

    image = Image.new("RGB", (img_width, img_height), color=border_color)
    draw = ImageDraw.Draw(image)

    for r in range(height):
        for c in range(width):
            color_idx = int(cells[r, c])
            fill_rgb = PALETTE.get(color_idx, (0, 0, 0))
            x0 = border_width + c * (cell_size + border_width)
            y0 = border_width + r * (cell_size + border_width)
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=fill_rgb)

    return image

def render_full_puzzle(
    task_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    cell_size: int = 28
) -> Path:
    """Renders all demonstration and test pairs cleanly onto an expanded, high-DPI canvas."""
    task_file = Path(task_path)
    with open(task_file, "r", encoding="utf-8") as f:
        task = json.load(f)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    title_font = get_font(size=18)
    label_font = get_font(size=13)
    arrow_font = get_font(size=16)

    columns = []

    # 1. Render Demonstration Columns
    for idx, pair in enumerate(train_pairs):
        in_img = render_grid_image(np.array(pair["input"]), cell_size=cell_size)
        out_img = render_grid_image(np.array(pair["output"]), cell_size=cell_size)

        col_w = max(in_img.width, out_img.width, 180) + 20
        col_h = 35 + in_img.height + 35 + out_img.height + 25

        col_canvas = Image.new("RGB", (col_w, col_h), color=(245, 247, 250))
        draw = ImageDraw.Draw(col_canvas)

        in_shape = f"{np.array(pair['input']).shape[0]}x{np.array(pair['input']).shape[1]}"
        out_shape = f"{np.array(pair['output']).shape[0]}x{np.array(pair['output']).shape[1]}"

        draw.text((10, 8), f"Train {idx + 1} Input ({in_shape})", fill=(30, 30, 30), font=label_font)
        col_canvas.paste(in_img, ((col_w - in_img.width) // 2, 35))

        arrow_y = 35 + in_img.height + 6
        draw.text(((col_w // 2) - 4, arrow_y), "↓", fill=(100, 100, 100), font=arrow_font)

        out_y = arrow_y + 26
        draw.text((10, out_y - 18), f"Output ({out_shape})", fill=(30, 30, 30), font=label_font)
        col_canvas.paste(out_img, ((col_w - out_img.width) // 2, out_y))

        columns.append(col_canvas)

    # 2. Render Test Challenge Column
    for idx, pair in enumerate(test_pairs):
        in_img = render_grid_image(np.array(pair["input"]), cell_size=cell_size)
        in_shape = f"{np.array(pair['input']).shape[0]}x{np.array(pair['input']).shape[1]}"

        if "output" in pair:
            out_img = render_grid_image(np.array(pair["output"]), cell_size=cell_size)
            out_shape = f"{np.array(pair['output']).shape[0]}x{np.array(pair['output']).shape[1]}"
            out_label = f"Target Output ({out_shape})"
        else:
            out_img = Image.new("RGB", in_img.size, color=(220, 220, 220))
            ImageDraw.Draw(out_img).text((out_img.width // 2 - 6, out_img.height // 2 - 8), "?", fill=(50, 50, 50), font=title_font)
            out_label = "Target Output (?)"

        col_w = max(in_img.width, out_img.width, 180) + 20
        col_h = 35 + in_img.height + 35 + out_img.height + 25

        col_canvas = Image.new("RGB", (col_w, col_h), color=(235, 240, 252))  # Distinct blue-gray accent
        draw = ImageDraw.Draw(col_canvas)

        draw.text((10, 8), f"TEST {idx + 1} Input ({in_shape})", fill=(180, 20, 20), font=label_font)
        col_canvas.paste(in_img, ((col_w - in_img.width) // 2, 35))

        arrow_y = 35 + in_img.height + 6
        draw.text(((col_w // 2) - 4, arrow_y), "↓", fill=(180, 20, 20), font=arrow_font)

        out_y = arrow_y + 26
        draw.text((10, out_y - 18), out_label, fill=(180, 20, 20), font=label_font)
        col_canvas.paste(out_img, ((col_w - out_img.width) // 2, out_y))

        columns.append(col_canvas)

    # 3. Stitch Columns Horizontally
    gap = 18
    title_height = 55
    total_w = sum(c.width for c in columns) + gap * (len(columns) + 1)
    max_h = max(c.height for c in columns) + title_height + 20

    sheet = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))
    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.text((22, 16), f"ARC-AGI-2 Task: {task_file.stem}", fill=(20, 20, 20), font=title_font)

    curr_x = gap
    for c in columns:
        sheet.paste(c, (curr_x, title_height))
        curr_x += c.width + gap

    if output_path is None:
        out_dir = Path("rendered_sheets")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{task_file.stem}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    sheet.save(output_path, dpi=(300, 300))
    return output_path