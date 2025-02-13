import argparse

from metric.rose_FENICE import RoSEFENICE
from metric.rose_FENICE_batched import RoSEFENICEBatched


def main():
    parser = argparse.ArgumentParser(description='RoSE Dataset FENICE Evaluator')
    parser.add_argument('--input', type=str, required=True, help='Path to RoSE dataset JSON')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--use_coref', action='store_true', help='Enable coreference resolution')
    parser.add_argument('--batch_size', type=int, default=16, help='Processing batch size')

    args = parser.parse_args()

    # Initialize evaluator with same params as original FENICE
    evaluator = RoSEFENICEBatched(
        use_coref=args.use_coref,
        nli_batch_size=args.batch_size,
        claim_extractor_batch_size=1,  # Disabled
        coreference_batch_size=args.batch_size
    )

    evaluator.process_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
