#!/bin/bash

small_side=512
cliplen_sec=15
max_tries=5
fps=30  # Set the desired frame rate here

data_dir="/mnt/SV_storage/VFM/EK100/EPIC-KITCHENS/"
save_dir="/mnt/SV_storage/VFM/EK100/EK100_512resolution"
# find all the subject folders that start with P
subjects=$(find $data_dir -mindepth 1 -maxdepth 1 -type d -name "P*")
subjects=( $subjects )  # to array
for subject_dir in "${subjects[@]}"; do
    # set the video dir as the subject folder with the videos folder
    indir="${subject_dir}/videos"
    outdir="${save_dir}/$(basename $subject_dir)"
    mkdir -p $outdir

    cd $indir || exit
    all_videos=$(find . -iname "*.MP4")
    all_videos=( $all_videos )  # to array
    cd -

    for video in "${all_videos[@]}"; do
        W=$( ffprobe -v quiet -show_format -show_streams -show_entries stream=width "${indir}/${video}" | grep width )
        W=${W#width=}
        H=$( ffprobe -v quiet -show_format -show_streams -show_entries stream=height "${indir}/${video}" | grep height )
        H=${H#height=}
        # Set the smaller side to small_side
        # from https://superuser.com/a/624564
        if [ $W -gt $H ] && [ $H -gt ${small_side} ]; then
            scale_str="-filter:v scale=-1:${small_side}"
        elif [ $H -gt $W ] && [ $W -gt ${small_side} ]; then
            scale_str="-filter:v scale=${small_side}:-1"
        else
            # The small side is smaller than required size, so don't resize/distort the video
            scale_str=""
        fi
        vidlen_sec=$( ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${indir}/${video}" )
        mkdir -p "${outdir}/${video}"
        for st_sec in $(seq 0 ${cliplen_sec} ${vidlen_sec}); do
            outfpath=${outdir}/${video}/${st_sec}.mp4
            try=0
            while [ $try -le $max_tries ]; do
                ffmpeg -y -ss ${st_sec} -i "${indir}/${video}" ${scale_str} -t ${cliplen_sec} -r ${fps} "${outfpath}"
                try=$(( $try + 1 ))
                write_errors=$( ffprobe -v error -i "${outfpath}" )
                # If no errors detected by ffprobe, we are done
                if [ -z "$write_errors" ]; then
                    echo $outfpath written successfully in $try tries!
                    break
                fi
            done
        done
        echo "Converted ${video}"
    done
done