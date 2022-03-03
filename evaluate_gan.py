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
    print(f"n: {n}, val_sum: {val_sum}, avg: {avg}")
    return deviation


def make_plot(dbase_name, feature_name, feature_units, dbase, start_run, runs, epochs):
    for i, run in enumerate(runs):
        plt.plot(epochs, run, f"-o", label=f"run{i + start_run}")
    plt.plot(epochs, (dbase, )*len(epochs), "k", label=dbase_name)
    plt.ylabel(f"{feature_name}({feature_units})")
    plt.xlabel("epochs")
    cur_ylim = plt.gca().get_ylim()
    plt.ylim(cur_ylim[0], cur_ylim[1] + 0.1)
    plt.gcf().legend(framealpha=1)
    plt.savefig(f"gan_results/plots/{dbase_name} runs, {feature_name}")
    plt.show()


def main():
    task = 0

    # show not collapse
    if task == 0:
        for run in range(1, 7):
            if run <= 3:
                folder = "battle"
            else:
                folder = "title"
            epoch = 3500
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
            print(f"notearrs: \n {notearrs} \n")

            single_note = False
            note_to_analyze = 0
            """
            create three types of histograms:
            - one for time deltas
            - one for note deltas
            - one for note length
            each histogram is for a single note across multiple melodies.
            compute the standard deviation of each histogram, and average all the standard deviations of the same type. 
            """

            grid = (3, 2)
            g_size = grid[0] * grid[1]
            val_names = {0: "time-delta", 1: "note-delta", 2: "duration"}

            all_deviations = []
            for i in range(3):
                fig, plots = plt.subplots(*grid, figsize=(10, 10))
                plot_i = 0

                if not single_note:
                    all_vals = notearrs[:, i::3]
                else:
                    all_vals = np.array([notearrs[:, note_to_analyze * 3 + i], ]).T
                print(f"val {i}: \n {all_vals} \n")
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
                all_deviations.append(avg_val_deviation)
                print(avg_val_deviation)

                plot_name = f"run{run}, epoch{epoch}, histograms of {val_names[i]} " \
                            f"over the first {g_size} notes of {len(files)} melodies"
                fig.suptitle(plot_name)
                fig.text(0.5, 0.02, f"average standard deviation on 100 notes: {avg_val_deviation}", ha="center")
                plt.savefig(f"gan_results/plots/r{run}e{epoch}g{g_size}_{val_names[i]}")
            print(all_deviations)

    # show improvement
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
                    epoch_avg = note_len[i:i+from_each_epoch]
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

    # show not running off
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
                epoch_files = list([file for file in files if f"h{epoch}" in file])[:16]
                for file in epoch_files:
                    print(file)
                    mfile = MidiFile(f"{path}/{file}")
                    ran_off = False
                    for msg in reversed(mfile.tracks[0][1:-1]):
                        if msg.note == 127 or msg.note == 0:
                            print("ran off")
                            epoch_count += 1
                            break
                run_count.append(epoch_count*100/from_each_epoch)
            all_count.append(run_count)
        print(all_count)

        np.savetxt("gan_results/plots/runoff_p.csv", np.asarray(all_count), delimiter=",")

        avg_count = []
        c_len = len(all_count)
        for i in range(len(saved_epochs)):
            c_sum = sum([x[i] for x in all_count])
            avg = c_sum/c_len
            avg_count.append(avg)
        plt.plot(saved_epochs, avg_count, f"-o", label=f"run average")
        plt.legend()
        plt.show()




