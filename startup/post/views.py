from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm

# Create your views here.
from rest_framework import generics
from .models import Post, Style, ReviewD
from .serializers import PostSerializer

from django.db import connection

import os 
#from dotenv import load_dotenv
import openai

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from glob import glob
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain
#import gdown
import os
from django.http import JsonResponse
import requests
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain.agents.agent import BaseSingleActionAgent


from langchain.document_loaders.csv_loader import CSVLoader

from PIL import Image
import pandas as pd
import shutil
import googletrans
from bs4 import BeautifulSoup
import openpyxl 

import selenium
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from django.conf.urls.static import static

import csv


def selecttext(request):
    select_text = request.GET.get('select_text')
    
    
    openai.api_key = "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"
    #os.environ['OPENAI_API_KEY'] = "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"

    # 모델 - GPT 3.5 Turbo 선택
    model = "gpt-3.5-turbo"

    # 질문 작성하기
    query = "'"+select_text+"'와 유사한 단어를 5개 추천해서 콤마로 구분해줘"

    # 메시지 설정하기
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]

    # ChatGPT API 호출하기
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages
    )
    answer = response['choices'][0]['message']['content']
    answer = answer.replace(' ','')
    list = answer.split(',')
    

    #translator = googletrans.Translator()
    #listJa = translator.translate(list, dest='ja').text
    #listCn = translator.translate(list, dest='zh-cn').text
    #listEn = translator.translate(list, dest='en').text

    print(answer)
    print(list)

    #prompt = input("'"+select_text+"'와 유사한 단어를 5개 추천해줘")
    #reponse = openai.Completion.create(
    #     model = "text-davinci-003",
    #     prompt = prompt,
    #     temperature = 1, 
    #     max_tokens = 1000,
    # )
    
    # selecttexts = reponse["choices"][0]["text"].strip()
    # print(reponse["choices"][0]["text"].strip())


    
    
    data = {
        "selecttexts": list
        
    }
    
    return JsonResponse(data) 



def main(request):
    message = request.GET.get('abc')
    print(message)

    return HttpResponse("신미림")

def image(request):
    #stylDetail = request.GET.get('stylDetail')
    styl_cd = request.GET.get('styl_cd')
    stylInfo = Style.objects.filter(styl_cd=styl_cd).values()
    print(stylInfo)
    translator = googletrans.Translator()
    result = translator.translate(stylInfo[0]['category'], dest='en')


    openai.api_key = "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"

    # for i in range(4):
    #    response = openai.Image.create(
    #     prompt=result.text+"The previous content is an explanation of clothes. Show me the picture of the person wearing the clothes I described",
    #     n=1,
    #     size="1024x1024",
    #     response_format="url"
    # )
    img_path=[]
    for i in range(4):
        response = openai.Image.create(
            #prompt=result.text+"The previous content is an explanation of clothes. Show me the picture of the person wearing the clothes I described",
            prompt="best picture of a "+result.text+"-wearing model for use at the shopping mall.",
            n=1,
            size="1024x1024",
            response_format="url"
        )
        image_url = response["data"][0]["url"]
        im = Image.open(requests.get(image_url, stream=True).raw)
        im.save('/Users/sinmilim/Documents/GitHub/DoBestStartUp/startup/frontend/src/pages/image'+str(i)+'.jpg')
        img_path.append('image'+str(i)+'.jpg')
        
    
    
    #im.show() #사진창이 뜸

 
    print(img_path)

    data = {
        "img_path":img_path
    }

    return JsonResponse(data)

def review(request):
    styl_cd = request.GET.get('styl_cd')
    os.environ['OPENAI_API_KEY'] = "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"
    
    reviewInfo = ReviewD.objects.filter(styl_cd=styl_cd).values()
    #print(reviewInfo)
    reviewList = []
    for review in reviewInfo:
        reviewList.append(review.get('reviewdetail'))
    
    
   
    doc_chunks = []
    file_name = "review.txt" 
    for line in reviewList:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, # 최대 청크 길이
            separators=["\n"], #  텍스트를 청크로 분할하는 데 사용되는 문자 목록
            chunk_overlap=0, # 인접한 청크 간에 중복되는 문자 수
        )
        chunks = text_splitter.split_text(line)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": i, "source": file_name}
            )
            doc_chunks.append(doc)

    #print(doc_chunks)vi
    print(doc_chunks)
    embeddings = OpenAIEmbeddings()
    index = Chroma.from_documents(doc_chunks, embeddings)

    
    system_template="""#To answer the question at the end, use the following context. If you don't know the answer, just say you don't know and don't try to make up an answer.
    #I want you to act as someone to summarize the explanation. Please write close to 400 letters to summarize the review.
    #you only answer in Korean

    #{summaries}
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)


    chain_type_kwargs = {"prompt": prompt}
    bk_chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0), 
        chain_type="stuff", 
        retriever=index.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True
    )

    result = bk_chain({"question": '리뷰를 한글로 공백 포함해서 300자이상 500자내로 요약해줘. 요약 시, 부정적인 키워드나 색깔, 디자인, 옷에 대한 특징이 빠지지 않게 해줘. '})

    print(f"질문 : {result['question']}")
    print()
    print(f"답변 : {result['answer']}")
    review_result = result['answer']
    translator = googletrans.Translator()
    review_result2 = translator.translate(result['answer'], dest='ja').text
    review_result3 = translator.translate(result['answer'], dest='zh-cn').text
    review_result4 = translator.translate(result['answer'], dest='en').text

    data = {
        "review_result":review_result
        ,"review_result2":review_result2
        ,"review_result3":review_result3
        ,"review_result4":review_result4
    }
    return JsonResponse(data)

    return HttpResponse(result['answer'])


def product(request):
    styl_cd = request.GET.get('styl_cd')
    mnfg_dtl_cntn1 = request.GET.get('mnfg_dtl_cntn1')
    mnfg_dtl_cntn2 = request.GET.get('mnfg_dtl_cntn2')
    mnfg_dtl_cntn3 = request.GET.get('mnfg_dtl_cntn3')
    
    stylInfo = Style.objects.filter(styl_cd=styl_cd).values()
    print(stylInfo)
    
    os.environ['OPENAI_API_KEY'] = "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"

    loader = CSVLoader(file_path='/Users/sinmilim/Desktop/product_total.csv')
    data = loader.load()

    #print(data)


    file_name = "product.csv"

    output = []
    # text 정제
    for page in data:
        text = page.page_content
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)   # 안녕-\n하세요 -> 안녕하세요  
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip()) # "인\n공\n\n지능펙\n토리 -> 인공지능펙토리
        text = re.sub(r"\n\s*\n", "\n\n", text) # \n버\n\n거\n\n킹\n -> 버\n거\n킹
        output.append(text)

    #print(output)

    doc_chunks = []

    for line in output:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, # 최대 청크 길이
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], #  텍스트를 청크로 분할하는 데 사용되는 문자 목록
            chunk_overlap=0, # 인접한 청크 간에 중복되는 문자 수
        )
        chunks = text_splitter.split_text(line)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": i, "source": file_name}
            )
            doc_chunks.append(doc)

    #print(doc_chunks)

    embeddings = OpenAIEmbeddings()
    index = Chroma.from_documents(doc_chunks, embeddings)

    #프롬프트 설정 이상한 부분이 나오지 않도록 설정 
    system_template="""To answer the question at the end, use the following context. If you don't know the answer, just say you don't know and don't try to make up an answer.
    I want you to act as my product explainer. 
    Below is an example.
    “Write a product description for shirt, TIME, cotton blend, S/S, seersucker material, back strap, stripe pattern, open collar design."

    you only answer in Korean
   
    {summaries}
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)


    chain_type_kwargs = {"prompt": prompt}
    bk_chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0), 
        chain_type="stuff", 
        retriever=index.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True
    )


    result = bk_chain({"question": ''+stylInfo[0]['category']+','+stylInfo[0]['brand_nm']+','+stylInfo[0]['material_cd']+','+stylInfo[0]['season_cd']+','+mnfg_dtl_cntn1+','+mnfg_dtl_cntn2+','+mnfg_dtl_cntn3+' 상품설명 써줘'})


    print(f"답변: {result['answer']}")
    stylDetail = result['answer'] 

    translator = googletrans.Translator()
    stylDetail2 = translator.translate(result['answer'], dest='ja').text
    stylDetail3 = translator.translate(result['answer'], dest='zh-cn').text
    stylDetail4 = translator.translate(result['answer'], dest='en').text

    
    data = {
        "stylDetail":stylDetail
        ,"styl_info2":stylDetail2
        ,"styl_info3":stylDetail3
        ,"styl_info4":stylDetail4
    }
    return JsonResponse(data)
    #return HttpResponse(stylDetail,img_path)##상품 
    # 
    # 


