"""Publication panels and complete contact sheets made only from recorded experimental outputs."""
from common import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def grid(cells, cols, path, tile=280):
    font = ImageFont.truetype(FONT, 16)
    small = ImageFont.truetype(FONT, 14)
    rows = (len(cells) + cols - 1) // cols
    canvas = Image.new(
        "RGB", (cols * (tile + 12) + 12, rows * (tile + 56) + 12), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for i, (image, label, extra) in enumerate(cells):
        x = 12 + (i % cols) * (tile + 12)
        y = 12 + (i // cols) * (tile + 56)
        im = Image.open(image).convert("RGB")
        im.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
        draw.text((x, y + tile + 3), label, font=font, fill="black")
        draw.text((x, y + tile + 25), extra, font=small, fill="#333333")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def samples(ch):
    return json.loads((ROOT / "results" / ch / "samples.json").read_text())


def main():
    figures = ROOT / "results/figures"
    figures.mkdir(exist_ok=True)
    # Chapter 2: fixed three seeds and two colors, without cherry-picking.
    folder = ROOT / "results/02_generation"
    rows = samples("02_generation")
    cells = []
    for color in ["red", "blue"]:
        for seed in [2026, 2027, 2028]:
            r = next(r for r in rows if r["kind"] == color and r["seed"] == seed)
            cells.append(
                (
                    folder / r["image"],
                    f"{color.title()} cup | seed {seed}",
                    f"output flag: {r['output_flag']}",
                )
            )
    grid(cells, 3, figures / "02_generation_pairs.png")
    # All 400 fairness outputs have contact sheets, with seed IDs and no classifier predictions.
    folder = ROOT / "results/07_fairness"
    rows = samples("07_fairness")
    for job in sorted({r["job"] for r in rows}):
        for mode in ["neutral", "female"]:
            selected = sorted(
                [r for r in rows if r["job"] == job and r["mode"] == mode],
                key=lambda r: r["seed"],
            )
            grid(
                [
                    (
                        folder / r["image"],
                        f"seed {r['seed']:02d}",
                        f"output flag: {r['output_flag']}",
                    )
                    for r in selected
                ],
                5,
                figures / "fairness_contact" / f"{job}_{mode}.png",
                tile=200,
            )
    selected = []
    for job in ["doctor", "nurse", "engineer", "teacher"]:
        for mode in ["neutral", "female"]:
            for seed in [0, 1]:
                r = next(
                    r
                    for r in rows
                    if r["job"] == job and r["mode"] == mode and r["seed"] == seed
                )
                selected.append(
                    (folder / r["image"], f"{job} | {mode}", f"seed {seed}")
                )
    grid(selected, 4, figures / "07_fairness_cases.png", tile=230)
    groups = json.loads((ROOT / "results/analysis_summary.json").read_text())[
        "fairness"
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d58b35", "#477bb8", "#b9b9b9", "#875c92"]
    labels = ["Feminine proxy", "Masculine proxy", "Uncertain", "Checker / review"]
    jobs = sorted({r["job"] for r in groups})
    x = np.arange(len(jobs))
    width = 0.37
    for offset, mode in [(-0.2, "neutral"), (0.2, "female")]:
        bottom = np.zeros(len(jobs))
        for field, color, label in zip(
            ["feminine", "masculine", "uncertain", "review"], colors, labels
        ):
            values = np.array(
                [
                    next(
                        r[field] for r in groups if r["job"] == j and r["mode"] == mode
                    )
                    for j in jobs
                ]
            )
            ax.bar(
                x + offset,
                values,
                width,
                bottom=bottom,
                color=color,
                label=label if mode == "neutral" else None,
                hatch="" if mode == "neutral" else "//",
            )
            bottom += values
    ax.set_xticks(x, jobs, rotation=35, ha="right")
    ax.set_ylim(0, 20)
    ax.set_ylabel("Recorded images (20 per condition)")
    ax.set_title("Neutral prompt (left) / explicit female prompt (right, hatched)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "07_fairness_counts.png", dpi=220)
    plt.close(fig)
    # Two-model/two-mode paired panels at fixed seed 2026, plus complete contact sheets.
    for ch, left, right, order in [
        (
            "06_esd",
            "base",
            "esd",
            ["target_landscape", "target_sunflowers", "control_cup", "control_forest"],
        ),
        ("06_sld", "off", "strong", ["swimwear", "clothed", "cup"]),
    ]:
        folder = ROOT / "results" / ch
        rows = samples(ch)
        cells = []
        for kind in order:
            for mode in [left, right]:
                r = next(
                    r
                    for r in rows
                    if r["kind"] == kind and r["mode"] == mode and r["seed"] == 2026
                )
                cells.append(
                    (
                        folder / r["image"],
                        f"{kind} | {mode}",
                        f"seed 2026 | flag: {r['output_flag']}",
                    )
                )
        grid(cells, 2, figures / (ch + "_pairs.png"), tile=320)
        grid(
            [
                (
                    folder / r["image"],
                    f"{r['kind']} | {r['mode']}",
                    f"seed {r['seed']} | flag: {r['output_flag']}",
                )
                for r in rows
            ],
            6,
            figures / (ch + "_all.png"),
            tile=200,
        )
    # CNNSpot actual scores for selected sources across JPEG conditions.
    folder = ROOT / "results/04_detection"
    rows = samples("04_detection")
    cells = []
    for ident in ["coffee", "red_2026", "landscape_2026"]:
        for quality in ["original", "100", "75", "40"]:
            r = next(r for r in rows if r["id"] == ident and r["condition"] == quality)
            cells.append(
                (
                    folder / r["image"],
                    f"{ident} | {quality}",
                    f"fake={r['fake_score']:.4g} | {r['prediction']}",
                )
            )
    grid(cells, 4, figures / "04_detection_cases.png", tile=250)
    # DCT recovery samples including crop failure; the decoded message is displayed verbatim.
    folder = ROOT / "results/05_watermark"
    rows = samples("05_watermark")
    cells = []
    for strength, condition in [
        (24, "unmarked"),
        (8, "png"),
        (24, "png"),
        (24, "jpeg40"),
        (24, "crop10"),
        (24, "resize256"),
    ]:
        r = next(
            r
            for r in rows
            if r["id"] == "red_2026"
            and r["strength"] == strength
            and r["condition"] == condition
        )
        cells.append(
            (
                folder / r["image"],
                f"{condition} | strength {strength}",
                f"{r['decoded_bits']} | errors {r['bit_errors']}/16",
            )
        )
    grid(cells, 3, figures / "05_watermark_cases.png", tile=300)
    matrix = []
    summary = json.loads((ROOT / "results/analysis_summary.json").read_text())[
        "watermark"
    ]
    conditions = [
        "png",
        "jpeg100",
        "jpeg75",
        "jpeg40",
        "resize256",
        "crop10",
        "noise2",
        "unmarked",
    ]
    for strength in [8, 16, 24]:
        matrix.append(
            [
                next(
                    r["bit_accuracy"]["mean"]
                    for r in summary
                    if r["strength"] == strength and r["condition"] == c
                )
                for c in conditions
            ]
        )
    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(8), conditions, rotation=25, ha="right")
    ax.set_yticks(range(3), [8, 16, 24])
    ax.set_ylabel("Embedding strength")
    for i in range(3):
        for j in range(8):
            ax.text(
                j,
                i,
                f"{matrix[i][j]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[i][j] > 0.65 else "black",
            )
    fig.colorbar(im, ax=ax, label="Mean bit recovery (9 images)")
    fig.tight_layout()
    fig.savefig(figures / "05_watermark_recovery.png", dpi=220)
    plt.close(fig)
    # Benign pseudo-word: one prescribed search seed per strategy, plus the natural reference.
    folder = ROOT / "results/03_pseudoword"
    rows = samples("03_pseudoword")
    chosen = [
        next(r for r in rows if r["mode"] == m and r["search_seed"] == 2026)
        for m in ["random", "improve"]
    ] + [next(r for r in rows if r["mode"] == "natural_reference")]
    grid(
        [
            (folder / r["image"], r["mode"], f"seed 2026 | flag {r['output_flag']}")
            for r in chosen
        ],
        3,
        figures / "03_pseudoword_generation.png",
        tile=320,
    )
    # Privacy: one prescribed synthetic reference, all input transformations and actual customized outputs.
    folder = ROOT / "results/08_privacy"
    rows = samples("08_privacy")
    ident = "face_2026"
    cells = []
    for variant in ["original", "protected", "protected_q75"]:
        r = next(
            r
            for r in rows
            if r["id"] == ident and r["stage"] == "feature" and r["variant"] == variant
        )
        cells.append(
            (
                folder / r["image"],
                variant,
                f"FaceNet cosine={r['cosine_to_original']:.4f}",
            )
        )
    for variant in ["original", "protected", "protected_q75"]:
        r = next(
            r
            for r in rows
            if r["id"] == ident
            and r["stage"] == "customization"
            and r["variant"] == variant
            and r["seed"] == 2026
        )
        sim = r["cosine_to_original"]
        cells.append(
            (
                folder / r["image"],
                "custom | " + variant,
                "FaceNet cosine=" + ("unscorable" if sim is None else f"{sim:.4f}"),
            )
        )
    grid(cells, 3, figures / "08_privacy_cases.png", tile=300)
    write_json(
        figures / "figure_manifest.json",
        {p.name: sha(p) for p in figures.glob("*.png")},
    )


if __name__ == "__main__":
    main()
