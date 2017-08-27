# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

names = []
for i in range(18):
    names.append('student' + str(i + 1))

gender = pd.Series(np.random.randint(0,2,18)).transform(lambda x: 'Male' if x == 1 else 'Female').as_matrix()
grade = np.random.randint(4,7, size= 18)
math = np.random.randint(65,89, size=18)
english = np.random.randint(65,89, size=18)
science = np.random.randint(65,89, size=18)

def trans_rating(score):
    if(score > 80):
        return 'good'
    elif(score > 70 and score <= 80):
        return 'fair'
    else:
        return 'average'
        

#rating = pd.Series(np.random.randint(0,3,18)).transform(lambda x: trans_rating(x)).as_matrix()

students = pd.DataFrame(dict(student_id=names, gender=gender, grade=grade,math=math,science=science,english=english))

avg_score = students[['math','science','english']].mean(axis = 1)

students['rating'] = avg_score.apply(trans_rating)

# specifying columns preserves this order in teh file
students.to_csv('D:\Project\Learning\MachineLearning\Git\MachineLearning\Pandas\students.csv', index=False,
                columns=['student_id', 'gender', 'grade', 'math', 'science', 'english','rating'])