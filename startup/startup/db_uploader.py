#임포트
import os
import sys
import csv
import django

#환경변수 세팅(뒷부분은 프로젝트명.settings로 설정한다.)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'startup.settings')
django.setup()

from post.models import *

REVIEW_PATH = '/Users/sinmilim/Desktop/review_test6.csv'


def insert_Menu():
    with open(REVIEW_PATH) as csv_file:
        data_reader = csv.reader(csv_file)
        next(data_reader, None)
        for row in data_reader:
            if row[0]:
                
                styl_cd = row[0]
                reviewdetail = row[1]
                ReviewD.objects.create(styl_cd = styl_cd, reviewdetail = reviewdetail)
    print('MENU DATA UPLOADED SUCCESSFULY!')