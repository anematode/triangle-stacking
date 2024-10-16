--save-state pipe.save -i ./pipe/MagrittePipe.jpg --json ./pipe/pipe.json -o ./pipe/final_pipe.png --intermediate pipe/pipe_frames

cat $(find . -maxdepth 1 -name '*.png' -print | sort -V) | ffmpeg -framerate 60 -i - -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p fast.mp4