# ~ upload_songs.py ~

### process_files():
+ Looks in current directory
+ searches for m4a and lrc files
+ extracts file metadata with ffmpeg
+ uses add_song_to_lookup() to add the song and logs
+ saves song_list.py after all songs done

### upload_and_track_files():
+ uploads to box and gets file_id

# ~ Server.py ~

### read_cut_config()
+ reads the cut_config() file or uses defaults

## --mode gencuts
### create_or_update_cut_list()
+ calls backup_cut_list()
    + backs up cut_list.csv into bak/
+ calls pull_song_list_from_box()
    + deletes current song_list.csv if found
    + pulls song_list from we-sharp/ on box
+ loads into song_df
+ loads/initializes cut_list into cut_df
+ iterates through song_df:
    + calls generate_cut_times() on each song to get cut times
    + adds or updates value in the cut_df
+ saves cut_df out when done
+ calls sync_with_box()
    + maybe call backup_cut_list() here?
    + deletes remote copy if exists (should have already backed up at this point)
    + uploads to we-sharp folder
    + sync backups?

## --mode preprocess
### preprocess_audio()






## TODO: 
+ handle uploads where the file may already exist
    -- log, skip and move to ./delete
+ folder id's for all folders?
+ want to back up cut list first before deleting and save backups
+ download cut_list.csv if it exists
+ separate dirs based on genre

-> reads cut_config

        Start_offset=30
        length=10
        n_cuts=3



we-sharp/

  |--/rock/ 
        |--song_list.csv
        |--cut_list.csv
        |--orig/ *.m4a, *.lrc
        |--dbo/ *wav
        |--other/ *wav
        |--vocals/ *wav
        |--drums/ *wav
        |--bass/ *wav
  |--/edm/ 
        |--song_list.csv
        |--cut_list.csv
        |--orig/ *.m4a, *.lrc
        |--dbo/ *wav
        |--other/ *wav
        |--vocals/ *wav
        |--drums/ *wav
        |--bass/ *wav
  |
  |--/bak/ cut_list_*.csv, song_list_*.csv


ffmpeg -i .m4a -ac 1 -ar 44100 -sample_fmt s16p output.wav 


grab_song()

extract 30 sec segment with ffmpeg:

    ffmpeg -ss 38 -t 30 -i output_dbo.wav -c copy output_sample.mp3
        get start time from cut col