import argparse
import timeit
from src.utils import setup_dbqa
import sys
import os 
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    # add parent path to python path so that we can import from src
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    start = timeit.default_timer() # Start timer

    # Replace with your custom model of choice
    model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # print(os.getcwd())

    '''
       pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100
    )
    #pipe.save_pretrained('../models/llm/')
    
    '''
 
    pipe = pipeline(
        "text2text-generation",
        model =  AutoModelForSeq2SeqLM.from_pretrained(os.getcwd()+'/models/llm/'),
        tokenizer = AutoTokenizer.from_pretrained(os.getcwd()+'/models/llm/',local_files_only=True),
        temperature=0.01,
        max_new_tokens=200
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    # Setup QA object
    dbqa = setup_dbqa(local_llm,os.getcwd())
    
    # Parse input from argparse into QA object
    response = dbqa({'query': args.input})
    end = timeit.default_timer() # End timer

    # Print document QA response
    print(f'\nRunning QA model')
    print(f'\nAnswer: {response["result"]}')
    print('='*20) # Formatting separator
       
    # Display time taken for CPU inference
    print(f"Time to retrieve response: {end - start}")