from argparse import ArgumentParser
import utility


def main():
    # load args
    in_args = load_args()
    # load the model
    model = utility.load_checkpoint(in_args.checkpoint_path, in_args.gpu, train=False)
    # predict
    utility.sanityChecking(
        in_args.image, in_args.category_names, model, in_args.gpu, in_args.top_k)


def load_args():

    parser = ArgumentParser(description='Train the model')

    parser.add_argument('image', type=str,
                        help='Path of flower image to predict')
    parser.add_argument('checkpoint_path', type=str,
                        help='checkpoint model')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top KK most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Return top KK most likely classes')
    parser.add_argument('--gpu', action='store_true',
                        default=False,
                        dest='gpu',
                        help='Use GPU for inference, set a switch to true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
