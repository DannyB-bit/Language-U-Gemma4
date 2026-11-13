"""Use the full decoded neural bank to propose and execute an arithmetic plan."""
import argparse
import json
from pathlib import Path

from neural_planner import solve


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--question',required=True)
    parser.add_argument('--chunk-out',type=Path)
    args=parser.parse_args()
    try:
        answer=solve(args.question,args.model.read_bytes())
        raw=answer.pop('packed_chunk')
        if args.chunk_out:
            with args.chunk_out.open('xb') as out:out.write(raw)
        answer['chunk_bytes']=len(raw)
        answer['external_question_utf8_bytes']=len(args.question.encode())
        print(json.dumps({'status':'answered',**answer},indent=2))
    except ValueError as error:
        print(json.dumps({'status':'rejected','reason':str(error)},indent=2))
        raise SystemExit(1)


if __name__=='__main__':main()
