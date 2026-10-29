"""Write an exploratory TLS-pattern report from saved spatial analysis outputs."""
import base64
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'whole-slide-20261012'/'tls_analysis'


def main():
    import pandas as pd
    summary=json.loads((OUT/'spatial_summary.json').read_text())
    regions=pd.read_csv(OUT/'ranked_candidate_regions.csv')
    null=summary['random_label_nulls']
    observations={
        'A1':'Compact, dense lymphoid-appearing aggregate. Predicted B/T cells are substantially intermixed.',
        'A2':'Dense lobulated lymphoid-appearing aggregate. Predicted B fraction is higher in the center than the surrounding annulus.',
        'A3':'Several adjacent lymphoid-appearing groups; less discrete than A1/A2.',
        'A4':'Lymphoid-appearing aggregation with surrounding infiltration; a stronger predicted B-core gradient but lower aggregate fraction.',
        'A5':'Smaller, less circumscribed infiltrate. Lower priority; the peak disappears at the broadest smoothing scale.',
        'A6':'Smaller localized infiltrate with a strong radial B gradient but limited predicted B numbers; sensitive to confidence and scale settings.'}
    regions['visual_impression_not_pathologist_annotation']=regions.candidate_id.map(observations)
    regions['suggested_review_priority']=regions.candidate_id.map({'A1':'first','A2':'first','A3':'second','A4':'second','A5':'lower','A6':'lower'})
    regions['TLS_confirmed']=False
    regions['TLS_maturity']='unassessed'
    regions.to_csv(OUT/'ranked_candidate_regions.csv',index=False)
    coordinates=[]
    for r in regions.itertuples():
        coordinates.append({'id':r.candidate_id,'center_level0_px':[r.x_level0_px,r.y_level0_px],
                            'center_um':[r.x_um,r.y_um], 'analysis_radius_um':150,
                            'review_window_um':600,'interpretation':'candidate for review; TLS not confirmed'})
    (OUT/'candidate_coordinates.json').write_text(json.dumps(coordinates,indent=2)+'\n')
    (OUT/'visual_review_notes.json').write_text(json.dumps({'reviewer':'AI visual impression; no pathologist confirmation',
                                                        'regions':observations},indent=2)+'\n')
    lines=['# Spatial patterns related to TLS in the example slide','',
           '**Main finding:** the predicted B and T cells form localized aggregates. Six regions pass an exploratory mixed-aggregate screen; A1 and A2 are the strongest initial H&E review priorities. These are candidate regions, not six confirmed TLS.','',
           '## Why these patterns are relevant','',
           'A primary NSCLC study described intratumoral lymphoid structures with dendritic-cell/T-cell clusters adjacent to B-cell follicles. This supports looking for local B/T organization rather than treating all lymphocyte-rich tissue as TLS. [Dieu-Nosjean et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18802153/)','',
           'TLS composition and maturity are distinct questions. A later NSCLC study assessed T cells, B cells, and follicular dendritic-cell features using CD3/CD8, CD20, CD21/CD23, and DC-LAMP. The current four-class H&E model does not provide those maturation markers. [TLS scoring study, 2024](https://pubmed.ncbi.nlm.nih.gov/39376565/)','',
           '## Patterns observed in this slide','',
           f"1. **Predicted B cells cluster at short distances.** Median B–B nearest-neighbor spacing is {summary['observed_median_BB_nearest_um']:.1f} µm. It averages {null['global_random_label_median_BB_um']['mean']:.1f} µm when B/T labels are shuffled across the same lymphocyte coordinates, and {null['local_250um_block_label_median_BB_um']['mean']:.1f} µm when labels are shuffled within local 250-µm blocks. Both one-sided empirical p-values are 0.005 (199 permutations; minimum attainable p=0.005). This tests predicted-label clustering, not TLS presence.",
           '2. **B cells have excess nearby B cells.** Within 25 µm of a predicted B cell, 33.9% of predicted lymphocyte neighbors are B, compared with 18.1% slide-wide. The fraction declines with distance, consistent with local rather than uniform B enrichment.',
           '3. **The strongest regions are mixed B/T aggregates.** A1–A3 have B+T fractions of 48–52% of detected cells, compared with 20.4% across the slide. Their B fractions among all detections are about 2.9–3.3 times the slide background. H&E shows dense lymphoid-appearing aggregates at A1 and A2.',
           '4. **Some regions show a central B enrichment with more T cells around it.** At A2, B cells constitute 31.5% of predicted B+T cells within 75 µm, versus 16.4% in the 75–250-µm annulus. The core/annulus B:T odds ratio is 2.36. A4 has a stronger ratio (3.18), but less concentrated overall infiltration. These gradients suggest organization worth reviewing; they are not evidence of a confirmed B follicle or germinal center.',
           '5. **B density alone is insufficient.** Eighteen B-density peaks were found at the default scale. Several nonpassing peaks are B-enriched/T-poor and appear gland-rich on H&E rather than showing a compact lymphoid aggregate. They should not be automatically called TLS.','',
           '## Candidate regions','',
           'Counts refer to fixed 150-µm-radius analysis disks, not histologically segmented TLS boundaries. Coordinates are measured from the upper-left of the original slide.','',
           '| Region | Center x,y (µm) | Predicted B | Predicted T | B+T / all cells | B enrichment | Initial priority |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for r in regions.itertuples():
        lines.append(f'| {r.candidate_id} | {r.x_um:.0f}, {r.y_um:.0f} | {r.B_150um} | {r.T_150um} | {100*r.lymphocyte_fraction_150um:.1f}% | {r.B_enrichment_vs_slide:.2f}× | {r.suggested_review_priority} |')
    lines+=['', '## H&E impressions for review','']
    for label,note in observations.items():
        lines.append(f'- **{label}:** {note}')
    lines+=['', 'These visual impressions are exploratory and are not pathologist annotations. No region was assigned a confirmed TLS, immature-TLS, or mature-TLS label.','',
            '## Sensitivity and uncertainty','',
            '- A1–A4 have nearby B-density peaks at Gaussian scales of 50, 100, and 150 µm, and when considering confidence cutoffs of 0.5, 0.7, and 0.9. Their peaks also persist using only B predictions unchanged after both horizontal and vertical image/mask flips.',
            '- Absolute support decreases at stricter cutoffs. A1, A2, A3, and A4 retain only 11, 15, 17, and 11 predicted B cells at confidence ≥0.9, respectively. Nearby-peak persistence is not evidence that these cell labels are correct.',
            '- Across the original whole-slide predictions, only 49.6% of B labels and 73.7% of T labels remain unchanged under both flips. In A1–A4, the corresponding stable B counts are 84, 82, 73, and 46. Confidence and stability are complementary checks, neither a biological gold standard.',
            '- A5/A6 are weaker findings: both disappear at the broadest smoothing scale, and A6 does not retain a matched peak at confidence ≥0.7 or ≥0.9.',
            '- Changing the heuristic screen thresholds produces 3–10 candidate regions. Therefore the output is a review shortlist, not a measured TLS count.',
            f"- The analysis excludes {summary['possible_repeat_detections_excluded']} likely overlap-repeat detections for spatial calculations: different patches, centroids within 2 level-0 pixels (0.504 µm), and bounding-box IoU ≥0.5; the highest-confidence member is retained. The original prediction CSV remains unchanged. An audit lists the excluded IDs.",
            '- Adjacent candidate disks may represent parts of one larger infiltrate. The heatmap is not a TLS segmentation and does not estimate TLS area or diameter.',
            '- These are patterns in model-assigned cell labels. Correlated model errors caused by local staining, morphology, or tissue type can also create spatial structure. The local permutation control cannot eliminate that possibility.','',
            '## How the distribution model was analyzed','',
            'The analysis uses physical-coordinate marked points, Gaussian density fields, and fixed-radius neighborhoods. Cell coordinates were converted with the slide’s 0.252-µm pixel spacing. Density fields use 25-µm bins. The default B field is smoothed with Gaussian σ=100 µm, and peaks are selected using a 17-bin nonmaximum filter. The minimum smoothed peak height is 0.2 B cells per grid bin; in confidence-filtered comparisons it scales with retained B numbers, with a floor of 0.02. Thus peak matching checks position, not a fixed absolute B density.',
            '', 'A region passes the exploratory mixed-aggregate screen if its 150-µm disk contains ≥30 predicted B cells, ≥100 predicted T cells, and a B+T fraction ≥0.30. These thresholds were selected for this exploratory screen and have no validated diagnostic meaning. No classifier was trained to call TLS. No individual candidate p-values are reported, because selecting centers from B peaks would bias such tests.',
            '', 'The nearest-neighbor null test fixes the observed predicted B/T positions and preserves their class counts. The local null additionally preserves B/T counts in 250-µm spatial blocks. Shorter observed B–B spacing than both nulls indicates nonuniform model-assigned B marks even after controlling much of the local lymphocyte-density pattern. It does not show patient-level associations, prognosis, or treatment response.','',
            '## What would distinguish TLS from a nonspecific infiltrate','',
            'Start with a pathologist review of A1/A2, then A3/A4, using the saved H&E regions and original-slide coordinates. If paired stains are available, CD3/CD20 can check the predicted T/B compartments, and CD21/CD23 can help assess follicular dendritic-cell organization and maturity, following the marker approach used in the NSCLC TLS scoring study. [Primary study](https://pubmed.ncbi.nlm.nih.gov/39376565/)','',
            'The current data do not establish germinal centers, high endothelial venules, follicular dendritic-cell networks, or T-cell subtypes. “Other” combines untyped cells, so it cannot define tumor versus stroma or the invasive margin. Tumor-compartment annotations are needed before testing whether candidates favor a tumor boundary. No survival or clinical response association can be inferred from this one slide.','',
            '## Saved outputs','',
            '- `TLS_spatial_dashboard.png`: candidate map, density fields, clustering test, radial composition, and confidence/stability comparison.',
            '- `candidate_HE_gallery.png` and `A1_HE.png` / `A1_BT_overlay.png` etc.: H&E and predicted B/T cells for manual review.',
            '- `ranked_candidate_regions.csv` and `candidate_coordinates.json`: region metrics and original-slide positions.',
            '- `spatial_summary.json`, `random_label_nulls.csv`, and `heuristic_threshold_sensitivity.csv`: reproducible metrics, null controls, and screen sensitivity.',
            '- `spatial_cells.csv` and `possible_overlap_repeat_audit.csv`: analysis points and conservative repeat-removal audit.',
            '- `all_predicted_BT_flip_checks.csv`: checks of all original predicted B/T cells.',
            '- `literature_sources.json`: links to primary research retrieved from PubMed; the built-in search tools were unavailable.',
            '', '![TLS-related spatial patterns](TLS_spatial_dashboard.png)', '', '![Candidate H&E review gallery](candidate_HE_gallery.png)']
    (OUT/'TLS_pattern_report.md').write_text('\n'.join(lines)+'\n')
    def embed(name):
        return '<img alt="'+html.escape(name)+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'
    display=regions[['candidate_id','x_um','y_um','B_150um','T_150um','lymphocyte_fraction_150um','B_enrichment_vs_slide','suggested_review_priority']].to_html(index=False,float_format=lambda v:f'{v:.3f}')
    narrative='\n'.join(line for line in lines if not line.startswith('!['))
    page='''<!doctype html><html><head><meta charset="utf-8"><title>TLS-related spatial patterns</title><style>body{max-width:1350px;margin:36px auto;padding:20px;font:16px system-ui;color:#182536}h1{font-size:30px}.note{padding:18px;background:#eef4fa;border-radius:12px}img{width:100%;height:auto;margin:18px 0}table{border-collapse:collapse;max-width:100%}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;line-height:1.6}.gallery{max-width:1100px}</style></head><body><h1>TLS-related spatial patterns</h1><p class="note">Six exploratory mixed B/T aggregate regions. A1/A2 are the first review priorities. No confirmed TLS count or maturity grade.</p>'''+embed('TLS_spatial_dashboard.png')+display+'<div class="gallery">'+embed('candidate_HE_gallery.png')+'</div><h2>Methods, results, and limitations</h2><pre>'+html.escape(narrative)+'</pre></body></html>'
    (OUT/'TLS_pattern_report.html').write_text(page)
    print('TLS pattern report and candidate review coordinates saved.')


if __name__=='__main__':
    main()
