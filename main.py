import find_classifier
import preparemidi
import createdata
import notearr
import halvemelodies
import gan
import evaluate_classifier
import evaluate_gan
import rand_for_surv


if __name__ == '__main__':
    PATH = f"D:/AlphaProject/_PythonML/MidiGAN"
    action = 9
    if action == 1:
        preparemidi.main()
    elif action == 2:
        createdata.main()
    elif action == 3:
        find_classifier.main()
    elif action == 4:
        halvemelodies.main()
    elif action == 5:
        notearr.main()
    elif action == 6:
        gan.main()
    elif action == 7:
        evaluate_classifier.main()
    elif action == 8:
        evaluate_gan.main()
    elif action == 9:
        rand_for_surv.main()



