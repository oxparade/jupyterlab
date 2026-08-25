from pathlib import Path


def export_plotly(fig, output_path_dir: Path, filename: str):
    fig.write_html(
        output_path_dir / f"{filename}.html",
        include_plotlyjs="cdn",
    )
    fig.write_image(output_path_dir / f"{filename}.png")
