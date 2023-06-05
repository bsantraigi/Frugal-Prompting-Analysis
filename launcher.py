import random
import wandb
import argparse

def cmdline_args(x=None):
    """Argument Parser
    --dataset: Dataset to use (MSC or TC)
    --model: Model to use (flan-t5, tk-instruct, T0)
    --prompt_type: "manual" or "ppl"
    --few_shot: Use 1 example per along with each query, otherwise zero_shot
    --background_knowledge: Use background knowledge or not (in summarized format)
    --history_signal_type: Type of history signal (full, peg, bart, recent-k, semantic-k, none)
    --history_k: Number of utterances to use in recent-k or semantic-k
    --wandb: Use wandb or not
    """
    parser = argparse.ArgumentParser(description='Launcher')
    parser.add_argument('-d', '--dataset', type=str, default='MSC', help='Dataset to use (MSC or TC)', choices=["MSC", "TC"])
    parser.add_argument('-m', '--model', type=str, default='flan-t5', help='Model to use (flan-t5, tk-instruct, T0)', choices=["flan-t5", "T0", "tk-instruct"])
    parser.add_argument('-pt','--prompt_type',  type=str, default='manual', help='manual or ppl', choices=["manual", "ppl"])
    parser.add_argument('-fs','--few_shot',  action="store_true", help='Use 1 example per along with each query, otherwise zero_shot')
    parser.add_argument('-bw','--background_knowledge',  action="store_true", help='Use background knowledge or not (in summarized format)')
    parser.add_argument('-hst','--history_signal_type',  type=str, default='full', choices=['full', 'peg', 'bart', 'recent-k', 'semantic-k', 'none'],
                        help='Type of history signal (full, peg, bart, recent-k, semantic-k, none)')
    parser.add_argument('-hk','--history_k',  type=int, default=4, help='Number of utterances to use in recent-k or semantic-k', choices=[2,4,8,10])
    parser.add_argument('-w','--wandb',  action="store_true", help='Use wandb or not')
    if x is not None:
        return parser.parse_args(x)
    else:
        return parser.parse_args()

def load_dataset(dataset, prompt_type, few_shot, background_knowledge, history_signal_type, history_k):
    """Load dataset
    """
    if dataset == "MSC":
        

def main():
    args = cmdline_args()
    print(args)
    arg_dict = vars(args)
    if args.wandb:
        wandb.init(project="frugal-prompts", config=arg_dict)

    dataset = load_dataset(
        dataset = args.dataset,
        prompt_type = args.prompt_type,
        few_shot = args.few_shot,
        background_knowledge = args.background_knowledge,
        history_signal_type = args.history_signal_type,
        history_k = args.history_k
    )
    
    # Test: report some random results to wandb for now
    if args.wandb:        
        # BLEU	METEOR	ROUGE-1	ROUGE-2	ROUGE-L	BERTScore-p	BERTScore-r	BERTScore-F1	DEB	BLEURT	prompt_len output_len
        wandb.log({
            "BLEU": random.random(),
            "METEOR": random.random(),
            "ROUGE-1": random.random(),
            "ROUGE-2": random.random(),
            "ROUGE-L": random.random(),
            "BERTScore-p": random.random(),
            "BERTScore-r": random.random(),
            "BERTScore-F1": random.random(),
            "DEB": random.random(),
            "BLEURT": random.random(),
            "prompt_len": random.randint(1, 10),
            "output_len": random.randint(1, 100)
        })
        wandb.finish()

if __name__ == "__main__":
    main()
