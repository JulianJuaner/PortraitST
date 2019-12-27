parser.add_argument('--root', type=str, default='../datasets', help='root path of test images')
        parser.add_argument('--name', type=str, default='gaintest', help='name of the dataset')
        parser.add_argument('--outf', type=str, default='man_test', help='name of the dataset')
        parser.add_argument('--max_size', type=tuple, default=(1024, 1024), help='the maximum size of test images')

        # Training options
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--lr', type=float, default=1, help='learning rate')
        parser.add_argument('--iter_show', type=int, default=10, help='iters to show the midate results')
        parser.add_argument('--layers', type=str, default=partialLayer)
        parser.add_argument('--epoch', type=int, default=50)
        parser.add_argument('--start', type=int, default=30)

        # Weight parameters
        parser.add_argument('--alpha_3', type=float, default=0.5, help='layer preference of conv3 first term')
        parser.add_argument('--alpha_4', type=float, default=0.5, help='layer preference of conv4 first term')
        parser.add_argument('--beta_3', type=float, default=0.5, help='layer preference of conv3 second term')
        parser.add_argument('--beta_4', type=float, default=0.5, help='layer preference of conv4 second term')

        parser.add_argument('--iter', type=int, default=150, help='iterations of feed-forward and back-propagation')
        parser.add_argument('--gmin', type=float, default=0.3, help='lower bound clamp gain map')
        parser.add_argument('--gmax', type=float, default=10.0, help='upper bound clamp gain map')
        parser.add_argument('--gT', type=float, default=5e-8, help='balance two terms in the total loss')
