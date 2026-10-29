"""Render TLS candidate maps, spatial diagnostics, and H&E review regions."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "whole-slide-20261012" / "tls_analysis"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))


def main():
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    from PIL import Image, ImageDraw, ImageFont
    import openslide
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    summary = json.loads((OUT / "spatial_summary.json").read_text())
    candidates = pd.read_csv(OUT / "ranked_candidate_regions.csv")
    cells = pd.read_csv(OUT / "spatial_cells.csv")
    null = pd.read_csv(OUT / "random_label_nulls.csv")
    fields = np.load(OUT / "density_fields.npz")
    xy = cells[["x_um", "y_um"]].to_numpy()
    tree = cKDTree(xy)
    b = cells.celltype.eq("B").to_numpy()
    t = cells.celltype.eq("T").to_numpy()
    btree, ttree = cKDTree(xy[b]), cKDTree(xy[t])
    profiles = []
    with openslide.OpenSlide(summary["slide"]) as slide:
        w, h = slide.dimensions
        thumbnail = slide.get_thumbnail((1800,1800)).convert("RGB")
        thumbnail.save(OUT / "slide_HE_thumbnail.png")
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
        except OSError:
            font = ImageFont.load_default(size=19)
        canvas = Image.new("RGB", (1100, 70+len(candidates)*365), "#14202d")
        draw = ImageDraw.Draw(canvas)
        draw.text((15,12), "TLS-related review candidates: H&E and predicted B/T locations", font=font, fill="white")
        draw.text((15,40), "600 × 600 µm regions; white circle = 150 µm analysis radius; B green / T gold", font=font, fill="white")
        for number,row in enumerate(candidates.itertuples()):
            center = np.array([row.x_um,row.y_um])
            half = 300
            start = np.floor((center-half)/summary["microns_per_pixel"]).astype(int)
            size = int(np.ceil(600/summary["microns_per_pixel"]/slide.level_downsamples[1]))
            roi = slide.read_region(tuple(start),1,(size,size)).convert("RGB")
            roi.save(OUT / f"{row.candidate_id}_HE.png")
            annotated = roi.copy()
            local = ImageDraw.Draw(annotated)
            selected = cells.iloc[tree.query_ball_point(center,425)]
            factor = 1/(summary["microns_per_pixel"]*slide.level_downsamples[1])
            for cell in selected.itertuples():
                if cell.celltype not in {"B","T"}:
                    continue
                x = (cell.x_um-start[0]*summary["microns_per_pixel"])*factor
                y = (cell.y_um-start[1]*summary["microns_per_pixel"])*factor
                if 0 <= x < size and 0 <= y < size:
                    color = "#46ed83" if cell.celltype=="B" else "#ffd43b"
                    local.ellipse((x-2,y-2,x+2,y+2),fill=color)
            cc = (center/summary["microns_per_pixel"]-start)/slide.level_downsamples[1]
            radius=150*factor
            local.ellipse((cc[0]-radius,cc[1]-radius,cc[0]+radius,cc[1]+radius),outline="white",width=2)
            annotated.save(OUT / f"{row.candidate_id}_BT_overlay.png")
            y=70+number*365
            canvas.paste(roi.resize((330,330)),(10,y))
            canvas.paste(annotated.resize((330,330)),(350,y))
            lines=[f"{row.candidate_id}: ({row.x_um:.0f}, {row.y_um:.0f}) µm",
                   f"B {row.B_150um}; T {row.T_150um}",
                   f"B+T fraction of all cells: {row.lymphocyte_fraction_150um:.1%}",
                   f"B enrichment vs slide: {row.B_enrichment_vs_slide:.2f}×",
                   f"B median confidence: {row.B_median_confidence:.3f}",
                   f"B stable under both flips: {row.B_stable_both_flips}",
                   f"Core B/T odds vs annulus: {row.core_annulus_B_odds_ratio:.2f}×",
                   "Candidate region; TLS not confirmed"]
            for i,text in enumerate(lines):
                draw.text((700,y+10+32*i),text,font=font,fill="white")
            for radius in [25,50,75,100,150,200,250,300]:
                ids=tree.query_ball_point(center,radius); sub=cells.iloc[ids]
                nb=int(sub.celltype.eq('B').sum()); nt=int(sub.celltype.eq('T').sum())
                profiles.append({"candidate_id":row.candidate_id,"radius_um":radius,"B":nb,"T":nt,
                                 "B_fraction_among_BT":nb/max(1,nb+nt)})
        canvas.save(OUT / "candidate_HE_gallery.png")
    pd.DataFrame(profiles).to_csv(OUT / "candidate_radial_profiles.csv",index=False)
    extent=[0,w*summary["microns_per_pixel"]/1000,h*summary["microns_per_pixel"]/1000,0]
    plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans","figure.facecolor":"#f5f8fb",
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout="constrained")
    fig.suptitle("Predicted-cell spatial patterns related to TLS screening\nSix exploratory aggregate candidates; no confirmed TLS or maturity labels",fontsize=18)
    axes[0,0].imshow(thumbnail,extent=extent)
    axes[0,0].set(title="Candidate locations on H&E",xlabel="x (mm)",ylabel="y (mm)")
    for row in candidates.itertuples():
        axes[0,0].add_patch(Circle((row.x_um/1000,row.y_um/1000),.15,fill=False,color="#ef4c5d",lw=2))
        axes[0,0].annotate(row.candidate_id,(row.x_um/1000,row.y_um/1000),xytext=(7,5),textcoords="offset points",color="#8d1422",fontweight="bold")
    im=axes[0,1].imshow(fields['B'].T/25**2*1e6,extent=extent,cmap="magma")
    axes[0,1].set(title="Predicted B-cell density (σ = 100 µm)",xlabel="x (mm)",ylabel="y (mm)")
    fig.colorbar(im,ax=axes[0,1],label="Smoothed predicted B cells / mm²",shrink=.7)
    fraction=(fields['B']+fields['T'])/np.maximum(fields['all'],1e-12)
    fraction=np.ma.masked_where(fields['all']<1,fraction)
    im=axes[0,2].imshow(fraction.T,extent=extent,cmap="YlGnBu",vmin=0,vmax=.65)
    axes[0,2].set(title="Predicted B+T fraction of detected cells",xlabel="x (mm)",ylabel="y (mm)")
    fig.colorbar(im,ax=axes[0,2],label="Local predicted lymphocyte fraction",shrink=.7)
    axes[1,0].hist(null.global_random_label_median_BB_um,bins=20,alpha=.65,label="Whole-slide label shuffle",color="#487bbb")
    axes[1,0].hist(null.local_250um_block_label_median_BB_um,bins=20,alpha=.65,label="Local 250 µm label shuffle",color="#50b18e")
    axes[1,0].axvline(summary['observed_median_BB_nearest_um'],color="#d54b55",lw=2,label="Observed")
    axes[1,0].set(title="B–B nearest-neighbor clustering test",xlabel="Median nearest-B distance (µm)",ylabel="Permutations")
    axes[1,0].legend(fontsize=8)
    for label,group in pd.DataFrame(profiles).groupby('candidate_id'):
        axes[1,1].plot(group.radius_um,group.B_fraction_among_BT,marker='.',label=label)
    axes[1,1].axhline(summary['analysis_counts']['B']/(summary['analysis_counts']['B']+summary['analysis_counts']['T']),color='gray',ls='--',label='Slide background')
    axes[1,1].set(title="Cumulative B fraction around each center",xlabel="Radius from candidate center (µm)",ylabel="B / (B+T)",ylim=(0,.8))
    axes[1,1].legend(ncol=3,fontsize=8)
    x=np.arange(len(candidates)); width=.25
    axes[1,2].bar(x-width,candidates['B_conf_ge_0.7']/candidates.B_150um,width,label='Confidence ≥0.7')
    axes[1,2].bar(x,candidates['B_conf_ge_0.9']/candidates.B_150um,width,label='Confidence ≥0.9')
    axes[1,2].bar(x+width,candidates.B_stable_both_flips/candidates.B_150um,width,label='Same B label after both flips')
    axes[1,2].set_xticks(x,candidates.candidate_id)
    axes[1,2].set(title="Candidate B-cell sensitivity",ylabel="Fraction of original predicted B cells",ylim=(0,1))
    axes[1,2].legend(fontsize=8)
    fig.savefig(OUT / "TLS_spatial_dashboard.png",dpi=150)
    plt.close(fig)
    print('Maps, radial profiles, and H&E candidate gallery saved.')


if __name__ == '__main__':
    main()
