import argparse

def get_args(description='Efficient Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")
    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")


    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    # parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    # parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')
    parser.add_argument('--pretrained_path', type=str, default="/home/ubuntu/.cache/clip", help='pretrained model path')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_thread_reader', default=2, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--clip_lr', type=float, default=6e-4, help='learning rate')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--start_val_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--feature_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cuda', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    parser.add_argument("--local-rank", default=0, type=int, help="distribted training")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=50, help='Information display frequence')
    parser.add_argument("--output_dir", default="log/", type=str, help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    
    parser.add_argument('--lora_dim', type=int, default=8)

    parser.add_argument('--shared_latent_space', type=str, default='transformer', choices=['transformer', 'linear'])

    ## add for unified prompt transformer and global visual prompts
    parser.add_argument('--text_prompt_length', type=int, default=4)
    parser.add_argument('--local_each_frame_prompt_length', type=int, default=4)
    parser.add_argument('--global_visual_prompt_length', type=int, default=8)
    parser.add_argument('--visual_output_type', type=str, default='global_token0', choices=['global_token0', 'average_global_token','average_frame_cls_token','average_global_token_and_frame_cls_token'],
                            help="The choice of visual output for abalation")
    parser.add_argument('--sel_layer', type=str, default='12')
    parser.add_argument('--frame_per_seg', type=str, default='2-4-12')
    parser.add_argument('--seg_num', type=str, default='6-3-1')
    parser.add_argument('--seg_layer', type=str,default='8-9-10')
    parser.add_argument('--r', type=int, default=10)
    parser.add_argument('--select', type=str, default='Dual-Attention', choices=['Dual-Attention', 'random', 'STA'])
    parser.add_argument('--alpha', type=float, default=0)

    # post-processing
    parser.add_argument("--DSL", default=False, type=bool, help="whether using dual softmax in post testing")
    args = parser.parse_args()

    return args