import mido as md
import os
import time
import random as rnd
import mido.midifiles.meta
import mergemid


# takes all midi files in in_path, and makes merged midi files in out_path. n is the amount of files to merge.
def merge_tracks(in_path, out_path, index):
    file_name = os.listdir(in_path)
    file_name = sorted(file_name)[index]
    merge_file = open(f"{out_path}/{file_name}", 'w')
    merge_file.close()
    try:
        mergemid.main(f"{in_path}/{file_name}", f"{out_path}/{file_name}")
    except AssertionError:
        return None
    except EOFError:
        return None
    return file_name


# makes a MidiFile array out of all midi files in a folder
def file_to_midifile(file_path):
    print(file_path)
    try:
        return md.MidiFile(f"{file_path}")
    except EOFError:
        os.remove(f"{file_path}")
        return None
    except IOError:
        os.remove(f"{file_path}")
        return None
    except mido.midifiles.meta.KeySignatureError:
        os.remove(f"{file_path}")
        return None
    except IndexError:
        os.remove(f"{file_path}")
        return None


# filters midi files to only a tempo message followed by note_on messages
def only_note(midifile):
    # print_midi(midifile)
    if midifile is None:
        return None
    onlyn_midi = md.MidiFile()
    in_track = midifile.tracks[0]
    out_track = md.MidiTrack()
    tempo = 500000
    tempo_set = False
    for j, msg in enumerate(in_track):
        if msg.type == 'set_tempo' and not tempo_set:
            tempo = msg.tempo
            tempo_set = True
        elif msg.type == 'note_on':
            out_track.append(translate_time(in_track, j, out_track))
        elif msg.type == 'note_off':
            off_msg = translate_time(in_track, j, out_track)
            out_track.append(md.Message('note_on', channel=off_msg.channel, note=off_msg.note, velocity=0,
                                        time=off_msg.time))
    onlyn_midi.ticks_per_beat = midifile.ticks_per_beat
    print(tempo)
    out_track.insert(0, md.MetaMessage('set_tempo', tempo=tempo, time=0))
    onlyn_midi.tracks.append(out_track)
    return onlyn_midi


# removes all percussion from filtered midi files
def no_percussion(midifile):
    if midifile is None:
        return None
    nop_midi = md.MidiFile()
    in_track = midifile.tracks[0][1:]
    out_track = md.MidiTrack()
    out_track.append(midifile.tracks[0][0])
    print(midifile.tracks[0][0])
    for j, msg in enumerate(in_track):
        if msg.channel < 9:
            out_track.append(translate_time(in_track, j, out_track).copy(channel=0))
    nop_midi.tracks.append(out_track)
    nop_midi.ticks_per_beat = midifile.ticks_per_beat
    return nop_midi


# turns filtered midi files into midi files with only the highest note at each moment
def highest_pitch(midifile):
    if midifile is None:
        return None
    highp_midi = md.MidiFile()
    out_track = md.MidiTrack()
    in_track = midifile.tracks[0][1:]
    # adds the tempo message
    out_track.append(midifile.tracks[0][0])
    # message and message index
    h_msg = None
    h_msg_i = 0
    time_zero = False
    for msg_i, msg in enumerate(in_track):
        if msg.velocity > 0:
            if h_msg is None:
                h_msg = msg
                h_msg_i = msg_i
            if msg.note > h_msg.note:
                if msg.time == 0:
                    h_msg = msg
                    h_msg_i = msg_i
                else:
                    if time_zero:
                        out_track.append(h_msg.copy(time=0, velocity=64))
                        out_track.append(translate_time(in_track, msg_i, out_track).copy(note=h_msg.note,
                                                                                         velocity=0))
                        h_msg = msg
                        h_msg_i = msg_i
                        time_zero = False
                    else:
                        out_track.append(translate_time(in_track, h_msg_i, out_track).copy(velocity=64))
                        out_track.append(translate_time(in_track, msg_i, out_track).copy(note=h_msg.note,
                                                                                         velocity=0))
                        h_msg = msg
                        h_msg_i = msg_i
                        time_zero = True
        if msg.velocity == 0:
            if h_msg is not None:
                if msg.note == h_msg.note:
                    if time_zero:
                        out_track.append(h_msg.copy(time=0, velocity=64))
                        out_track.append(translate_time(in_track, msg_i, out_track))
                        h_msg = None
                        time_zero = False
                    else:
                        out_track.append(translate_time(in_track, h_msg_i, out_track).copy(velocity=64))
                        out_track.append(translate_time(in_track, msg_i, out_track))
                        h_msg = None
    highp_midi.tracks.append(out_track)
    highp_midi.ticks_per_beat = midifile.ticks_per_beat
    # print_midi(midi_arr[i])
    return highp_midi


def same_length(midifile, n):
    if midifile is None:
        return None
    samel_midi = md.MidiFile()
    out_track = md.MidiTrack()
    out_track.append(midifile.tracks[0][0])
    print(midifile.tracks[0][0])
    in_track = midifile.tracks[0][1:]
    if len(in_track) >= n*2:
        out_track += in_track[:n*2]
    else:
        if len(in_track) >= n:
            out_track += in_track
            out_track += in_track[:n*2-len(in_track)]
        else:
            return None
    samel_midi.tracks.append(out_track)
    samel_midi.ticks_per_beat = midifile.ticks_per_beat
    # print_midi(midi_arr[i])
    # print(len(midi_arr[i].tracks[0]))
    return samel_midi


def remove_ghost_notes(midifile):
    if midifile is None:
        return None
    noghost_midi = md.MidiFile()
    out_track = md.MidiTrack()
    in_track = midifile.tracks[0][1:]
    # adds the tempo message
    out_track.append(midifile.tracks[0][0])
    for msg in in_track:
        if msg.velocity > 0:
            last_msg = msg
        else:
            if not (msg.note == last_msg.note and msg.time <= 20):
                out_track.append(last_msg)
                out_track.append(msg)
    noghost_midi.tracks.append(out_track)
    noghost_midi.ticks_per_beat = midifile.ticks_per_beat
    return noghost_midi


# replaces all midi files in the path with midi files from the array
def replace_files(midifile, out_path, file_name):
    if midifile is None:
        os.remove(f"{out_path}/{file_name}")
        return
    midifile.save(f"{out_path}/{file_name}")


# redundant. finds the note off message for a given note on and gives it the correct delta time
def find_note_off(track, i):
    total_time = 0
    on_msg = track[i]
    while i < len(track):
        i += 1
        msg = track[i]
        total_time += msg.time
        if msg.type == 'note_on':
            if msg.velocity == 0:
                if msg.note == on_msg.note:
                    return msg.copy(time=total_time)
    print("no note_off found")
    return None


# calculates the delta time needed considering removed messages
def translate_time(in_track, i, out_track):
    total_time_in = 0
    total_time_out = 0
    for x in range(i+1):
        total_time_in += in_track[x].time
    for msg in out_track:
        total_time_out += msg.time
    out_msg = in_track[i].copy(time=(total_time_in-total_time_out))
    return out_msg


# redundant. prints midi files with an option for a message filter
def print_midi(midi, types=None, control=-1):
    for i, track in enumerate(midi.tracks):
        print(f'Track {i}: {track.name}')
        for msg in track:
            if types is None:
                print(msg)
            elif msg.type in types:
                if control >= 0:
                    if msg.control == control:
                        print(msg)
                else:
                    print(msg)

                    # break


def main(folder, index, amount_of_notes):
    # print(md.MidiFile('../rawmusic/title/1999.mid'))
    IN_PATH = '../rawmusic'
    OUT_PATH = 'melodies'
    start_time = time.time()
    file_name = merge_tracks(f"{IN_PATH}/{folder}", f"{OUT_PATH}/{folder}", index)
    if file_name is None:
        return
    print("Merged tracks")
    midifile = file_to_midifile(f"{OUT_PATH}/{folder}")
    print("Turned to arrays")
    midifile = only_note(midifile)
    print("Filtered notes")
    midifile = no_percussion(midifile)
    print("Filtered percussion")
    midifile = highest_pitch(midifile)
    print("Filtered highest pitch")
    midifile = same_length(midifile, amount_of_notes)
    print("Standardised length")
    midifile = remove_ghost_notes(midifile)
    midifile = same_length(midifile, amount_of_notes)
    print("Removed ghost notes")
    replace_files(midifile, f"{OUT_PATH}/{folder}")
    print(time.time() - start_time)


if __name__ == "main":
    pass
    # folder = 'test'
    # main(folder)

    # test file debugging
    # if True:
    #     t_file = md.MidiFile()
    #     t_track = md.MidiTrack()
    #     t_track.append(md.MetaMessage(type='set_tempo', tempo=480000, time=0))
    #     t_track.append(md.Message(type='note_on', note=62, time=150))
    #     t_track.append(md.Message(type='note_on', note=67, time=100))
    #     t_track.append(md.Message(type='control_change', time=150))
    #     t_track.append(md.Message(type='note_on', velocity=0, note=62, time=150))
    #     t_track.append(md.Message(type='note_on', velocity=0, note=67, time=0))
    #     t_file.tracks.append(t_track)
    #     print(t_file)
    #     t_file.save('test.mid')
    #     midifiles = [t_file]
    #     midifiles = only_note(midifiles)
    #     midifiles = highest_pitch(midifiles)
    #     t_file = midifiles[0]
    #     print(t_file)
    #     t_file.save('test2.mid')