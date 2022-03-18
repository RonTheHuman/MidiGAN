import mido as md
import os
from math import floor
import numpy as np


def folder_to_midifile_arr(in_path):
    midi_arr = []
    for file_name in sorted(os.listdir(in_path)):
        print(file_name)
        midi_arr.append(md.MidiFile(f"{in_path}/{file_name}"))
    return midi_arr


def midifile_arr_to_note_arrs_norm(midi_arr, default_tpb=48, default_mspb=500000):
    note_arrs = []
    for midifile in midi_arr:
        note_arr = list()
        in_track = midifile.tracks[0]
        conv_ratio = (in_track[0].tempo/midifile.ticks_per_beat) * (default_tpb/default_mspb)
        prev_note = 60
        for msg in in_track[1:-1]:
            if msg.velocity > 0:
                note_arr.append(floor(msg.time * conv_ratio))
                note_arr.append(msg.note - prev_note)
                prev_note = msg.note
            else:
                note_arr.append(floor(msg.time * conv_ratio))
        note_arrs.append(note_arr)
    return note_arrs


def note_arrs_to_midifile_arr_norm(note_arrs, default_tpb, default_mspb):
    midi_arr = []
    for note_arr in note_arrs:
        out_midi = md.MidiFile()
        out_track = md.MidiTrack()
        out_midi.ticks_per_beat = default_tpb
        out_track.append(md.MetaMessage('set_tempo', tempo=default_mspb))
        data_type = 0
        prev_note = 60
        """
        data types:
        0: ticks from prev note
        1: semitones from prev note
        2: duration in ticks 
        """
        for data in note_arr:
            data = data
            # print(data)
            if data_type == 0:
                on_time = data
                data_type += 1
            elif data_type == 1:
                prev_note = clamp(prev_note + data, 0, 127)
                out_track.append(md.Message('note_on', channel=0, note=prev_note, velocity=64, time=abs(on_time)))
                data_type += 1
            elif data_type == 2:
                out_track.append(md.Message('note_on', channel=0, note=prev_note, velocity=0, time=abs(data)))
                data_type = 0
        out_midi.tracks.append(out_track)
        midi_arr.append(out_midi)
    return midi_arr


# used in gan code when generating melodies
def save_midi(midifiles, path):
    for i, midifile in enumerate(midifiles):
        midifile.save(f"{path}_{i + 1}.mid")


def clamp(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def main():
    folders = ["battle", "title"]
    in_path = "melodies"
    out_path = "melody_arrays"
    for folder in folders:
        midifile_arr = folder_to_midifile_arr(f"{in_path}/{folder}")
        # print(repr(midifile_arr))
        note_arrs = midifile_arr_to_note_arrs_norm(midifile_arr, 48, 500000)
        print(np.array(note_arrs).shape)
        np.save(f"{out_path}/{folder}", np.array(note_arrs))
        with open(f"{out_path}/{folder}.txt", "w") as file:
            file.write(repr(note_arrs))
