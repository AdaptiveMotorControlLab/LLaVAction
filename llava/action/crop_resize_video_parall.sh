#!/usr/bin/env bash

small_side=512
cliplen_sec=15
max_tries=5
fps=30  # Set the desired frame rate
MAX_JOBS=15  # <-- Adjust this to control how many processes run in parallel

data_dir="/mnt/SV_storage/VFM/EK100/EPIC-KITCHENS/"
save_dir="/mnt/SV_storage/VFM/EK100/EK100_512resolution"

# Find all the subject folders that start with P
subjects=$(find "$data_dir" -mindepth 1 -maxdepth 1 -type d -name "P*")

# sort the subjects and start from subject P05
subjects=$(echo "$subjects" | sort -V | grep -A 1000 P05)

for subject_dir in $subjects; do

    indir="${subject_dir}/videos"
    outdir="${save_dir}/$(basename "$subject_dir")"
    mkdir -p "$outdir"

    # Gather all videos in this subject's "videos" folder
    cd "$indir" || exit
    all_videos=$(find . -iname "*.MP4")
    cd - > /dev/null

    for video in $all_videos; do

        # Extract width/height
        W=$(ffprobe -v quiet -show_format -show_streams -show_entries stream=width  "$indir/$video" \
            | grep width= | cut -d= -f2)
        H=$(ffprobe -v quiet -show_format -show_streams -show_entries stream=height "$indir/$video" \
            | grep height= | cut -d= -f2)

        # Decide scaling filter
        if [ "$W" -gt "$H" ] && [ "$H" -gt "$small_side" ]; then
            scale_str="-filter:v scale=-1:${small_side}"
        elif [ "$H" -gt "$W" ] && [ "$W" -gt "$small_side" ]; then
            scale_str="-filter:v scale=${small_side}:-1"
        else
            scale_str=""
        fi

        vidlen_sec=$(ffprobe -v error -show_entries format=duration -of \
                     default=noprint_wrappers=1:nokey=1 "$indir/$video")

        mkdir -p "${outdir}/${video}"

        # Generate clips
        for st_sec in $(seq 0 $cliplen_sec "${vidlen_sec%.*}"); do
            outfpath="${outdir}/${video}/${st_sec}.mp4"

            # Start a sub-shell { ... } in the background (&)
            {
                try=0
                while [ $try -le $max_tries ]; do
                    ffmpeg -y -ss "${st_sec}" -i "$indir/$video" \
                        $scale_str -t $cliplen_sec -r $fps \
                        "${outfpath}"

                    # Check if written successfully
                    write_errors=$(ffprobe -v error -i "$outfpath")
                    if [ -z "$write_errors" ]; then
                        echo "OK: ${outfpath} written successfully in $((try+1)) tries"
                        break
                    else
                        echo "ERROR writing ${outfpath}, retrying..."
                    fi
                    ((try++))
                done
            } &  # run in background

            # Limit concurrency
            while [ "$(jobs -p | wc -l)" -ge "$MAX_JOBS" ]; do
                # `wait -n` waits until one background job finishes.
                # (Available in Bash 4.3+; for older Bash, use `wait` without -n.)
                wait -n
            done

        done  # end of st_sec loop

    done  # end of videos loop

done  # end of subjects loop

# Wait for any remaining jobs still in flight
wait
echo "All conversions done."