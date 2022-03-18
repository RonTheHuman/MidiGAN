import mido as md
import os


def folder_to_midifile_arr(in_path):
    midi_arr = []
    for file_name in sorted(os.listdir(in_path)):
        print(file_name)
        midi_arr.append(md.MidiFile(f"{in_path}/{file_name}"))
    return midi_arr


def halve_files(midifiles):
    midi_arr = []
    for i, midifile in enumerate(midifiles):
        out_file_1 = md.MidiFile()
        out_file_2 = md.MidiFile()
        out_file_1.ticks_per_beat = midifile.ticks_per_beat
        out_file_2.ticks_per_beat = midifile.ticks_per_beat
        in_track = midifile.tracks[0]
        out_track_1 = md.MidiTrack()
        out_track_2 = md.MidiTrack()
        out_track_1.append(in_track[0])
        out_track_2.append(in_track[0])

        for msg in in_track[1:101]:
            out_track_1.append(msg)
        for msg in in_track[101:]:
            out_track_2.append(msg)

        out_file_1.tracks.append(out_track_1)
        out_file_2.tracks.append(out_track_2)

        midi_arr.append(out_file_1)
        midi_arr.append(out_file_2)
    return midi_arr


def save_midi(midifiles, path):
    for i, midifile in enumerate(midifiles):
        midifile.save(f"{path}_{i}.mid")


def main():
    IN_PATH = 'D:/AlphaProject/_PythonML/MidiGAN/melodies'
    folders = ["battle", "title"]
    for folder in folders:
        midi_files = folder_to_midifile_arr(f"{IN_PATH}/{folder}")
        midi_files = halve_files(midi_files)
        save_midi(midi_files, f"{IN_PATH}/{folder}_halved/{folder}_halved")
