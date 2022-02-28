import find_classifier
import preparemidi
import createdata
import notearr
import halvemelodies
import old_gan
import gan
import evaluate_classifier


if __name__ == '__main__':
    PATH = f"D:/AlphaProject/_PythonML/MidiGAN"
    action = 8
    if action == 1:
        # preparemidi.main('snes/normal', 0, 100)
        preparemidi.main('title', 1000, 100)
    elif action == 2:
        createdata.main()
    elif action == 3:
        find_classifier.main()
    elif action == 4:
        halvemelodies.main(["battle", "title"])
    elif action == 5:
        notearr.main()
    elif action == 6:
        old_gan.main()
    elif action == 7:
        gan.main()
    elif action == 8:
        evaluate_classifier.main()


