from pdf2image import convert_from_path 
import easyocr
import numpy as np
import PIL # Python Imaging Library
from PIL import ImageDraw, Image # drawing bounding boxes
import spacy # advanced NLP for key attributes

def createBoxes(bounds):
    categories =['publications','community service','experience','technicallmanagement skills',
             'project experience','activities','awards','summary','volunteer','education',
             'interests','skills','projects','work experience','professional experience','references',
             'volunteer experience','technical skills','research experience','computer skills',
             'leadership & volunteer experience','skillset','extracurriculars', 'certifications', 'certification',
             'academic projects','education & credentials','leadership and extra curriculars',
             'skills & certifications','skill set_','extra curricular activities',
             'leadership , awards & achievements','academic experience','leadership and achievements',
            'education and training','work history','professional summary','skills and abilities',
            'cqurse workand proiects','relevant skills','skill highlights','educations','experiences',
            'personality and language','related courses and skills','key skills','people & capability development',
            'skills and certifications', 'personal','caree interests','additional','interests 0','employment history',
             'details','relevant coursework','other','social impact','experience_', 'leadership and community engagement',
            'professional & leadership experience','additional skills/interests', 'leadership & volunteer experience',
            'other_','social impact','professional distinctions','additional information','activities and leadership',
            'additional experience','interests and skills','athletics','recent experience','awards and personal','selected patents and publications',
            'awards, honors, and interests','activities & leadership','leadership & community activities','additional info',
             'leadership','technical experience','community leadership & interests','community activities & additional information',
            'miscellaneous','leadership & community involvement','prior work experience','publication','personal','extracurricular experience',
            'additional skills and projects','publications & research','leadership, community & other','additional data',
            'community work','awards and honours','others','volunteering and public service','skills, interests & publications','personal interests',
            'languages','community & interests','community involement/personal','activities and skills','awards & community involvement',
            'entrepreneurial experience','entrepreneurship','media & technology experience','business ownership','service and interests',
            'extracurricular','certifications','skills & personal','other information','activities','professional sports experience',
            'other activities and personal interests','professional','writing & publications','skills/activities','community & other',
            'board experience','impact investing work','product, user, and strategy work','extra-curricular & community activities',
            'additional interests','additional data','additional experience','social entrepreneurship','interests and extracurriculars',
            'skills & personal','professional certifications and awards','community involvement','selected publications','volunteer experience and additional skills',
            'internship experience','employment','community engagement','awards, speaking engagements & press','leadership & other activities','other leadership experience',
            'hobbies','initiatives','additional projects','professional experience and leadership','professional experience & leadership','volunteer activities/activities outside job',
            'other inerests/hobbies','professional experiences','athletic experience','community','skills/additional information','education_','additional_','community leadership',
                'academic experience','academic experience_','prqeessional experience','leadershp & volunteer experience','additional:','volunteer & leadership experience',
                'dditional','additiona','ed uc a tio n','e xp e rie n c e','ad dttio nal','awards, honors; and interests',
            'professional experience: united states marine corps','skills and interests','leadership & additional information',
            'leadership , community & other','addtional leadershp','experience__','prqeessional_experience','honors & awards',
            'financial skills','extracurricular leadership','additional skills & interests','prqfessional experience','personal activities and interests',
            'leadership experience and service: collegiate activities','leadership experience and service: post-collegiate activities','professional expereince','honors and awards',
            'additional skills','skills & interests','leadershp & communty involvement','workexperience','addtional','honors; skills, & interests','leadership_awards_& skills',
            'skills &','awards &','personal and interests','leadership and activities','education:','work experience:','skills_hobbies & interests',
            'leadership experience','additional activties and interests','community_leadership_','interests & skills',
            'leadership & involvement','awards & interests','work','activities:', 'awards:','skills, achievements & interests','leadership & activities','additional leadership , skills, and interests',
            'education _','communty leadershp','skills, activities & interests','skills, languages and interests','experience (u.s_navy, submarines)','additional_experience','activities & interests',
            'skills and personal','leadership activities','professional experience:','leadership experience:','key skills:','e d u c a tio n','ex p e rienc e',
            'p e r s 0 na l','additional leadership','additional information and interests','professional experience_','leadership & service',
            'skills, activities and interests','selected publica tions','teaching','iternshps','public service','communty involvement',
            'professlonal experience','activities and interests','leadership & extracurricular','additional experience_','extracurricular activities & skills',
            'leadership & interests','leadership & extracurricular activittes','leadership and social impact','additional projects_',
            'education and honors','learn to_win (lzw executive and management experience_','naval intelligence officer_experience','navy surface warfare officer experience',
            'education & honors','leadership experience_','summary: strategic, results-oriented leader with experience building cross-functional systems and processes. looking to',
            'military','extracurricular activities','other experience','qther','edlcation','leadershpandcommunty service','education; honors and scholarships',
            'other interestsihobbies','volunteer activitiesiactivities outside job','leadership and additional information','skills/ additional information','extracurricular activities',
            'community and personal interests','community leadership & additional']
    box = []
    for x in bounds:
        if x[1].lower() in categories:
            box.append(x)
    box.append(x)
    return box

def giveProperNames(new_bounds):
    properNameBounds = []
    educationNames = ['education','education & credentials','academic experience','education and training','educations','education_','academic experience',
                     'academic experience_','ed uc a tio n','education:','education _','e d u c a tio n','education and honors','education & honors','edlcation','education; honors and scholarships']
    
    workNames = ['professional experience','work experience','experience','work history','experiences','experience_','recent work experience','prior work experience',
                'entrepreneurial experience','employment','professional experience and leadership','professional experience & leadership','professional experiences','recent experience',
                'media & technology experience','business ownership','additional experience','professional & leadership experience','prqeessional experience',
                'e xp e rie n c e','professional experience: united states marine corps','experience__','prqeessional_experience','prqfessional experience','professional expereince',
                'workexperience','work experience:','entrepreneurship','work','experience (u.s_navy, submarines)','professional experience:','ex p e rienc e','professional experience_',
                'iternshps','professlonal experience','learn to_win (lzw executive and management experience_','naval intelligence officer_experience','navy surface warfare officer experience']
    
    skillNames = ['skills','technicallmanagement skills','computer skills','skillset','skill set_','relevant skills',
              'skills and abilities','skill highlights','skills/additional information','skills/activities','skills & personal',
                 'skills/additional information', 'activities and skills', 'skills', 'skills, interests & publications',
                 'interests and skills', 'additional skills/interests','skills and interests','financial skills','additional skills & interests',
                  'additional skills','skills & interests','leadership_awards_& skills', 'skills &','skills_hobbies & interests','interests & skills',
                 'skills, achievements & interests','skills, activities & interests','skills, languages and interests','skills and personal',
                 'key skills:','skills, activities and interests','extracurricular activities & skills','skills/ additional information']
    for x in new_bounds:
        if(len(properNameBounds)==0):
            properNameBounds.append(x)
        elif(x[1] in educationNames):
            properNameBounds.append((x[0],'Education'))
        elif(x[1] in workNames):
            properNameBounds.append((x[0],'Work Experience'))
        elif(x[1] in skillNames):
            properNameBounds.append((x[0],'Skills'))        
        else:
            properNameBounds.append((x[0],'Extra'))
    return properNameBounds

def draw_boxes(image,bounds,color="yellow",width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0,p1,p2,p3 = bound[0]
        draw.line([*p0,*p1,*p2,*p3,*p0],fill=color,width=width)
    return image

def createNormalBounds(box):
    new_bounds = []
    last_section = 'Personal Info'
    for x in box:
        if len(new_bounds)==0:
            new_bounds.append(([[0, x[0][0][1]-10], [1700,x[0][0][1]-10], [1700, 0], [0,0]], last_section))
        elif len(new_bounds)==1:
            new_bounds.append(([[0,new_bounds[-1][0][0][1]+10],[1700,new_bounds[-1][0][0][1]+10],[1700,x[0][0][1]-10],[0,x[0][0][1]-10]], last_section))
        else:
            new_bounds.append(([[0,new_bounds[-1][0][3][1]+10],[1700,new_bounds[-1][0][3][1]+10],[1700,x[0][0][1]-10],[0,x[0][0][1]-10]],last_section))
        last_section = x[1].lower()
    new_bounds[-1][0][3][1]=2200
    new_bounds[-1][0][2][1]=2200
    return new_bounds

def createColumnBounds(box):
    new_bounds = []
    box.sort(key = lambda x: x[0][0][0])
    box.sort(key = lambda x: x[0][0][1])

    for i in range(2, len(box)):
        if len(new_bounds)==0:
            last_section = 'Personal Info'
            if (box[i-2][0][0][0]<500):
                new_bounds.append(([[0, 0], [1700,0], [1700, box[i-2][0][0][1]-10], 
                                [0,box[i-2][0][0][1]-10]], last_section))
                last_section = box[i-2][1].lower()
        if len(new_bounds)==1:
            new_bounds.append(([[0,box[i-2][0][0][1]],[1700,box[i-2][0][0][1]],
                                [1700,box[i-1][0][0][1]-10],[0,box[i-1][0][0][1]-10]], last_section))
            last_section = box[i-1][1].lower()
            new_bounds.append(([[0,box[i-1][0][0][1]],[1700,box[i-1][0][0][1]],[1700,box[i][0][0][1]-10],
                            [0,box[i][0][0][1]-10]],last_section))
            last_section = box[i][1].lower()
            print(new_bounds)
        if(box[i][0][0][0] <550):
            new_bounds.append(([[box[i][0][0][0],box[i][0][0][1]],[1700,box[i][0][0][1]],[1700,box[i][0][3][1]-10],
                        [box[i][0][0][0],box[i][0][3][1]-10]],last_section))

        elif(box[i][0][0][0]>550) and (box[i-1][0][0][0]<550):
            new_bounds[-1][0][3][1]=2200
            new_bounds[-1][0][2][1]=2200
            last_section=box[i][1].lower()
            endOfColumn = i
            break
    
   
    for i in range(endOfColumn, len(box)-1):
        for x in new_bounds:
            if box[i][0][2][1]<x[0][2][1]:
                x[0][1][0]=box[i][0][0][0]-10
                x[0][2][0]=box[i][0][0][0]-10
    
    for i in range(endOfColumn+1, len(box)):
        last_section=box[i-1][1].lower()
        new_bounds.append(([[box[i-1][0][0][0],box[i-1][0][0][1]],[1700,box[i-1][0][0][1]],[1700,box[i][0][0][1]-10],
                        [box[i-1][0][0][0],box[i][0][0][1]-10]],last_section))
    new_bounds[-1][0][3][1]=2200
    new_bounds[-1][0][2][1]=2200
    return new_bounds


    
