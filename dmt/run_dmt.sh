#!/usr/bin/bash

if [[ $1 == "-h" ]]
then
  echo "arguments: [njobs], [data_path], [input_path], [output_path], [dim (l, w, h)], [persistence_threshold] [mask_name]"
else
  njobs=$1
  data_path=$2
  input_path=$3
  output_path=$4
  len=$5
  wid=$6
  height=$7
  ph_thld=$8
  fig_name=$9

  dipha_path="dipha-graph-recon/build/dipha"
  dmt_path="src/build_dmt"

  # (1). Generate complex as dipha input
  python write_dipha_file_3d_revise.py $data_path ${input_path}complex.bin ${input_path}vert.txt

  # (2). Compute PH via dipha
  mpiexec -n $njobs $dipha_path ${input_path}complex.bin ${input_path}diagram.bin ${input_path}dipha.edges $len $wid $height

  # (3). Convert dipha vertices to .txt
  ./convert_dipha.py -i ${input_path}dipha.edges -o ${input_path}dipha_edges.txt

  # (4). Reconstruct DMT
  #for i in `seq 0 10 300`; do
  #  $dmt_path ${input_path}vert.txt ${input_path}dipha_edges.txt $i $output_path
  #  python visualize.py -f ${output_path}dimo_vert.txt -m ${output_path}${fig_name}_${i}.tiff -d $len $wid $height
  #done
  
  $dmt_path ${input_path}vert.txt ${input_path}dipha_edges.txt $ph_thld $output_path

  # (5). Visualize
  python visualize.py -f ${output_path}dimo_vert.txt -m ${output_path}${fig_name}.tiff -d $len $wid $height

fi
