import os
import numpy as np
import matplotlib.pyplot as plt
from mido import MidiFile
from notearr import midifile_arr_to_note_arrs_norm
from math import sqrt
import pandas


def make_hgram(vals):
    hgram = {}
    for val in vals:
        if val not in hgram:
            hgram[val] = 1
            continue
        hgram[val] += 1
    return hgram


def dict_deviation(dict):
    n = sum(dict.values())
    val_sum = sum([k * v for k, v in dict.items()])
    avg = val_sum / n
    sdist_sum = sum([((k - avg) ** 2) * v for k, v in dict.items()])  # squared distance from average sum
    deviation = sqrt(sdist_sum / n)
    # print(f"n: {n}, val_sum: {val_sum}, avg: {avg}")
    return deviation


def make_plot(dbase_name, feature_name, feature_units, dbase, start_run, runs, epochs):
    for i, run in enumerate(runs):
        plt.plot(epochs, run, f"-o", label=f"run{i + start_run}")
    plt.plot(epochs, (dbase,) * len(epochs), "k", label=dbase_name)
    plt.ylabel(f"{feature_name}({feature_units})")
    plt.xlabel("epochs")
    cur_ylim = plt.gca().get_ylim()
    plt.ylim(cur_ylim[0], cur_ylim[1] + 0.1)
    plt.gcf().legend(framealpha=1)
    plt.savefig(f"gan_results/plots/{dbase_name} runs, {feature_name}")
    plt.show()


def main():
    task = 3

    # show not collapse
    if task == 0:
        all_deviations = []
        for run in range(1, 7):
            if run <= 3:
                folder = "battle"
            else:
                folder = "title"
            epoch = 4096
            file_contains = f"h{epoch}"
            path = f"gan_results/conv_gan_1.1/{folder}/midi/run{run}"

            files = os.listdir(path)
            files = [f"{path}/{f}" for f in files if file_contains in f]
            if len(files) == 0:
                print("incorrect filename error")
                exit()
            print(f"files head: {files[:10]}")
            notearrs = np.array(midifile_arr_to_note_arrs_norm([MidiFile(midi) for midi in files]))
            # notearrs = np.array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
            #                      [2, 3, 4, 2, 3, 4, 2, 3, 4],
            #                      [3, 4, 5, 3, 4, 5, 3, 4, 5]])
            # print(f"notearrs: \n {notearrs} \n")

            single_note = False
            note_to_analyze = 1
            """
            create three types of histograms:
            - one for time deltas
            - one for note deltas
            - one for note length
            each histogram is for a single note across multiple melodies.
            compute the standard deviation of each histogram, and average all the standard deviations of the same type. 
            """

            grid = (1, 1)
            g_size = grid[0] * grid[1]
            val_names = ("time-delta", "note-delta", "duration")
            val_units = ("ticks", "semitones", "ticks")

            run_deviations = []
            for i in range(3):
                fig, plots = plt.subplots(*grid, figsize=(9, 6))
                plot_i = 0

                if not single_note:
                    all_vals = notearrs[:, i::3]
                else:
                    all_vals = np.array([notearrs[:, note_to_analyze * 3 + i], ]).T
                print(f"val {val_names[i]}:")
                # print(f" \n {all_vals} \n")
                val_deviations = []
                for note_vals in all_vals.T:
                    val_hgram = make_hgram(note_vals)
                    val_deviations.append(dict_deviation(val_hgram))

                    if plot_i < g_size:
                        sorted_hgram = {k: v for k, v in sorted(val_hgram.items(), key=lambda x: x[0])}
                        if g_size != 1:
                            coord = (plot_i % grid[0], int(plot_i / grid[0]))
                            plots[coord].bar([str(k) for k in sorted_hgram.keys()], sorted_hgram.values())
                        else:
                            plots.bar([str(k) for k in sorted_hgram.keys()], sorted_hgram.values())
                        plot_i += 1

                avg_val_deviation = np.average(val_deviations)
                run_deviations.append(avg_val_deviation)
                # print(val_deviations)
                print(avg_val_deviation)
                if single_note:
                    s = ""
                else:
                    s = "s"
                plot_name = f"run{run}, epoch{epoch}, histogram{s} of {val_names[i]} " \
                            f"over the first {g_size} note{s} of {len(files)} melodies"
                plt.title(plot_name)
                if single_note:
                    # fig.text(0.5, 0.015, f"average standard deviation on 1 note: {avg_val_deviation}", ha="center")
                    pass
                else:
                    fig.text(0.5, 0.015, f"average standard deviation on 100 notes: {avg_val_deviation}", ha="center")
                plt.xlabel(f"{val_names[i]} ({val_units[i]})")
                plt.ylabel("amount")
                plt.savefig(f"gan_results/plots/r{run}e{epoch}g{g_size}_{val_names[i]}")
            if single_note:
                exit()
            all_deviations.append(run_deviations)
        np.savetxt("gan_results/plots/over_final_epoch_deviations.csv", np.asarray(all_deviations), delimiter=",")
        print(all_deviations)

    # create average note length graph
    if task == 1:
        note_len_files = sorted(os.listdir("gan_results/features/note_length"))
        run_avgs = []
        saved_epochs = ("16", "50", "100", "300", "700", "1024", "2048", "2500", "3000", "3500")
        from_each_epoch = 16
        dbase_avgs = {}
        for file in note_len_files:
            print(file)
            note_len = pandas.read_csv(f"gan_results/features/note_length/{file}")
            note_len = note_len.drop(note_len.columns[0], axis=1)
            if "run" in file:
                epoch_avgs = []
                for i in range(len(saved_epochs)):
                    epoch_avg = note_len[i:i + from_each_epoch]
                    epoch_avg = epoch_avg.mean(0)[0]
                    epoch_avgs.append(epoch_avg)
                    print(file)
                run_avgs.append(epoch_avgs)
            else:
                dbase_avgs[file] = note_len.mean(0)[0]
        print(f"{run_avgs} \n \n {dbase_avgs}")

        make_plot("battle", "average note length", "sec",
                  dbase_avgs["battle.csv"], 1, run_avgs[:3], saved_epochs)
        make_plot("title", "average note length", "sec",
                  dbase_avgs["title.csv"], 4, run_avgs[3:], saved_epochs)

    # show not running off: reaching top or bottom note
    if task == 2:
        all_count = []
        saved_epochs = ("16", "50", "100", "300", "700", "1024", "2048", "2500", "3000", "3500")
        from_each_epoch = 16
        for run in range(1, 7):
            if run <= 3:
                folder = "battle"
            else:
                folder = "title"
            path = f"gan_results/conv_gan_1.1/{folder}/midi/run{run}"

            files = os.listdir(path)
            run_count = []
            for epoch in saved_epochs:
                epoch_count = 0
                # get [from_each_epoch] melodies from each epoch
                epoch_files = list([file for file in files if f"h{epoch}" in file])[:from_each_epoch]
                for file in epoch_files:
                    print(file)
                    mfile = MidiFile(f"{path}/{file}")
                    for msg in reversed(mfile.tracks[0][1:-1]):
                        if msg.note == 127 or msg.note == 0:
                            print("ran off")
                            epoch_count += 1
                            break
                run_count.append(epoch_count * 100 / from_each_epoch)
            all_count.append(run_count)
        print(all_count)

        np.savetxt("gan_results/plots/runoff_p_5.csv", np.asarray(all_count), delimiter=",")

        avg_count = []
        c_len = len(all_count)
        for i in range(len(saved_epochs)):
            c_sum = sum([x[i] for x in all_count])
            avg = c_sum/c_len
            avg_count.append(avg)
        plt.plot(saved_epochs, avg_count, f"-o")
        plt.xlabel("epochs")
        plt.ylabel("percentage of melodies running off")
        plt.savefig("gan_results/plots/avg_runoff_p")
        plt.show()

    # check num of highest notes in dbase
    if task == 3:
        all_count = 0
        for folder in ("title", "battle"):
            path = f"melodies/{folder}"
            files = os.listdir(path)
            for file in files:
                print(file)
                mfile = MidiFile(f"{path}/{file}")
                for msg in reversed(mfile.tracks[0][1:-1]):
                    if msg.note == 127 or msg.note == 0:
                        print("ran off")
                        all_count += 1
                        break
        print(f"all count: {all_count}")
