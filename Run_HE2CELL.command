#!/bin/zsh
# Double-click in Finder to select a slide and run the local pipeline.
set -u
project_dir="${0:A:h}"
cd "$project_dir"
print "H&E cell-type identification — Other, T, Macrophage, B"
print "Results will be saved in this project's outputs folder."
print "Results are archived after verification. Keep the original slide for cell-crop access."
slide_path=$(osascript -e 'POSIX path of (choose file with prompt "Select an H&E slide (.svs, .tif, or .tiff)")') || exit 0
# Same order as he2cell.py: user override, stain-robust fine-tune (default), original checkpoint.
classifier_path="$project_dir/models/best_model.pth"
for candidate in he2cell_esca_finetuned.pth best_model2.pth; do
  if [[ ! -f "$classifier_path" && -f "$project_dir/models/$candidate" ]]; then
    classifier_path="$project_dir/models/$candidate"
  fi
done
if [[ ! -f "$classifier_path" ]]; then
  classifier_path=$(osascript -e 'POSIX path of (choose file with prompt "Select your trained best_model.pth checkpoint")') || exit 0
fi
output_path="$project_dir/outputs/run_$(date +%Y%m%d_%H%M%S)"
"$project_dir/he2cell" run --input "$slide_path" --classifier "$classifier_path" --output "$output_path"
result_status=$?
if [[ $result_status -eq 0 ]]; then
  open "$output_path"
else
  print "The run failed. The error is shown above."
fi
read "?Press Return to close."
exit $result_status
