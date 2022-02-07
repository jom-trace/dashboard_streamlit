#from email import header
import json
import numpy as np
from pkg_resources import UnknownExtra
import streamlit as st
import pandas as pd
import networkx as nx
from apyori import apriori
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
import requests
from sklearn.cluster import KMeans
from sklearn import preprocessing
from pymongo import MongoClient
from dateutil import parser
st.set_page_config(layout="wide")
session = requests.Session()

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

xdata = fetch(session, f"https://jom-trace-backend.herokuapp.com/getPatients")
user_id = []
final_data=[]
unknown_data = []
location = []
read_token = []


for i in range(len(xdata)):
    user = xdata[i]["username"]
    token = xdata[i]["deviceToken"]
    user_id.append([xdata[i]["uuid"],xdata[i]["username"]])
    read_token.append([user,token])
    for x in range(len(xdata[i]["closeContact"])):
        cc = (xdata[i]["closeContact"][x]["_uuid"])
        #location = (xdata[i]["locationVisited"][x]["locationName"])
        date = (xdata[i]["closeContact"][x]["date"])     
        final_data.append([user,cc,date])
    
    for x in range(len(xdata[i]["locationVisited"])):
        loc = (xdata[i]["locationVisited"][x]["loc"])
        location.append([user,loc])


for i in range (len(final_data)):
    newDate=parser.parse(final_data[i][2])
    formatted_date = newDate.strftime("%Y/%m/%d")
    final_data[i][2]=formatted_date


# for i in range(len(user_id)):
#     for y in range(len(final_data)):
#         found = False
#         if (user_id[i][0]==final_data[y][1]):
#             if(found==True):
#                 found=True
#                 temp = user_id[i][1]
#                 final_data[y][1]=temp
    
#     temp="Unknown"
#     final_data[i][1]=temp


count=1
i = 0

while i < len(final_data):
    found=False
    for y in range (len(user_id)):
        if(final_data[i][1]==user_id[y][0]):
               temp=user_id[y][1]
               final_data[i][1]=temp
               found=True
    if(found==False):
        #temp=("Unknown " + str(count))
        count=count+1
        #final_data[i][1]=temp
        unknown_data.append(final_data[i])
        final_data.pop(i)
    else:
        i +=1

i = 0

while i < len(final_data):
     x = 0
     while x < len(final_data):
        print(x)
        if(final_data[i][0]==final_data[x][1]):
            if(final_data[i][1]==final_data[x][0]):
               if(final_data[i][2]==final_data[x][2]):   
                    final_data.pop(x)
                    break
        x+=1
     i+=1


cnt=0
i = 0

while i < len(location):
    cnt = 0
    for y in range (len(final_data)):
        if(location[i][0]==final_data[y][0]):
            cnt = cnt + 1
        if(location[i][0]==final_data[y][1]):
            cnt = cnt + 1
    if(cnt == 0):
        location.pop(i)
    else:
        i += 1
            



# while i < len(location):
#      x = 0
#      while x < len(unknown_data):
#         print(x)
#         if(location[i][0] == unknown_data[x][0]):
#             location.pop(x)
#             break
#         x+=1
#      i+=1




# for i in range (len(final_data)):   
#    for x in range(len(collected_data)):
#        if(final_data[i][0]!=collected_data[x][1]):
#             if(final_data[i][1]!=collected_data[x][0]):

# final_data.pop(2)

# for i in range(len(user)):
#    for x in range(len(cc)):
#        for y in range(len(location)):
#            for z in range(len(date)):
#               if(cc[x][1]==user[i][0]):
#                  if(user[i][0]==location[y][0]):
#                     if(user[i][0]==date[z][0]):
#                         final_data.append([user[i][1],cc[x][0],location[y][1],date[z][1]])

# for i in range(len(user)):
#    for x in range(len(cc)):
#        for y in range(len(location)):
#            for z in range(len(date)):
#               if(cc[x][1]==user[i][0]):
#                  if(user[i][0]==location[y][0]):
#                     if(user[i][0]==date[z][0]):
#                         final_data2.append([cc[x][0],user[i][1],location[y][1],date[z][1]])

def cs_body():
    col1, col2 = st.columns(2)
    col1.title('One Mode Analysis')
    col2.title('Two Mode Analysis')

    st.write(unknown_data)
    st.write(final_data)
    
    data = pd.DataFrame(final_data, columns=("From","To","Date"))
    data2= data.drop(['Date'], axis = 1)
    with col1:
        st.subheader("Social Network Graph")
        graph = nx.from_pandas_edgelist(data2,source="From", target="To")
        fig = nx.draw_kamada_kawai(graph, with_labels=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        #################################################################################################################################################
        st.text("")
        st.text("")
        st.text("")
        st.subheader("Apriori Analysis")
        data_list = []
        for i in range (0,5):
            data_list.append([str(data2.values[i,j]) for j in range(0,2)])
        association_rules = apriori(data_list, min_support=0.003, min_confidence=0.7, min_lift=1.3, min_length=2)
        association_results = list(association_rules)

        def inspect(association_results):
            lhs         = [tuple(result[2][0][0])[0] for result in association_results]
            rhs         = [tuple(result[2][0][1])[0] for result in association_results]
            support    = [result[1] for result in association_results]
            confidence = [result[2][0][2] for result in association_results]
            lift       = [result[2][0][3] for result in association_results]
            return list(zip(lhs, rhs, support, confidence, lift))

        output_DataFrame = pd.DataFrame(inspect(association_results), columns = ['Left_Hand_Side', 'Right_Hand_Side', 'Support', 'Confidence', 'Lift'])
        st.write(output_DataFrame)
        #################################################################################################################################################
        st.text("")
        st.text("")
        st.text("")
        
        st.subheader("Clustering in One Mode Analysis")
        st.write("Scatter Plot before Clustering")
        
        country_map = {country:i for i, country in enumerate(data2.stack().unique())}
        new_data=data2.copy()

        new_data['From'] = new_data['From'].map(country_map)    
        new_data['To'] = new_data['To'].map(country_map)
        plt.scatter(new_data["From"],new_data["To"])
        st.pyplot()

        st.write("Scatter Plot after Clustering")
        kmeans = KMeans(n_clusters=2)
        x_scaled = preprocessing.scale(new_data)
        y_predicted=kmeans.fit_predict(x_scaled)

        new_data2=pd.DataFrame(x_scaled, columns=list('xy'))
        new_data2['cluster'] = y_predicted
        df1 = new_data2[new_data2.cluster==0]
        df2 = new_data2[new_data2.cluster==1]
        df3 = new_data2[new_data2.cluster==2]

        plt.scatter(df1.x,df1["y"],color='green')
        plt.scatter(df2.x,df2["y"],color='red')
        plt.scatter(df3.x,df3["y"],color='black')
        plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

        plt.xlabel('From')
        plt.ylabel('To')
        plt.legend()
        st.pyplot()
        #################################################################################################################################################
     

    
    with col2:
        new_data3 = pd.DataFrame(final_data, columns=("From","To","Date"))
        new_data4 = pd.DataFrame(location,columns=("user","location"))
        st.subheader("Social Network Graph with Geography")
        people1 = list(new_data3['From'])
        people2 = list(new_data3['To'])
        place = list(new_data4['location'])
        people3 = list(new_data4['user'])

        def create_from_edgelist(top,bottom):
            B = nx.Graph()
            for i in range(len(top)):
                B.add_node(top[i],bipartite=0)
                B.add_node(bottom[i],bipartite=1)
                B.add_edge(top[i],bottom[i])
            return B
        B = create_from_edgelist(people1,people2)

        def create_from_edgelist2(top,bottom):
            for i in range(len(top)):
                B.add_node(top[i],bipartite=0)
                B.add_node(bottom[i],bipartite=2)
                B.add_edge(top[i],bottom[i])
            return B
        B = create_from_edgelist2(people3,place)

        f = plt.figure(1,figsize=(8,6),dpi=400)
        pos = nx.spring_layout(B)
        colors = {0:'y',1:'y',2:'r'}
        shapes = {0:"s",1:"d",2:"d"}
        nx.draw_kamada_kawai(B, with_labels = True,node_color=[colors[B.nodes[node]['bipartite']]for node in B])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        #################################################################################################################################################
        st.text("")
        st.text("")
        st.text("")
        st.subheader("Searching Close Contact People...")
        Name = ""
        user_input = st.text_input("Enter a Positive Patient's Name:", Name)

        if user_input:

            ple = []
            for i in range(len(people3)):
                ple.append([people3[i], place[i]])

            print(ple)

            ple2 = []
            for i in range(len(people1)):
                ple2.append([people1[i], people2[i]])

            people_cont = []
            place_cont = []
            warn = []
            no_warn = []
            token_cont = []
            counter = 0

            def warning(name):

                global counter 
                for i in range(len(ple2)):
                    if name == ple2[i][0]:
                        people_cont.append(ple2[i][1])
                    elif (name == ple2[i][1]):
                        people_cont.append(ple2[i][0])
    
                for i in range(len(ple)):
                    if name == ple[i][0]:
                        place_cont.append(ple[i][1])

                for i in range(len(people_cont)):
                    for j in range(len(place_cont)):
                        counter = 0
                        for x in range(len(ple)):
                            if(place_cont[j] == ple[x][1]):
                                if(people_cont[i] == ple[x][0]):
                                    counter = counter + 1
            
                    if counter > 0: 
                        warn.append(people_cont[i]) 
                    else:
                        no_warn.append(people_cont[i])

                for i in range(len(read_token)):
                    for x in range(len(warn)):
                        if(read_token[i][0] == warn[x]):
                            token_cont.append(read_token[i][1])
                       

                output_df = pd.DataFrame({'People Close Contact' : [warn],
                                'Not Close Contact' : [no_warn]}, columns = ['People Close Contact', 'Not Close Contact'])
                st.write(output_df)
                st.button('Send Warning')
                if st.button:
                    url = 'https://jom-trace-backend.herokuapp.com/pushExposure'
                    myobj = {'closeContact': token_cont, 'messageTitle': 'Warning Message', 'messageBody': 'Stay Safe'}
                    headers = {
                    'Content-Type': 'application/json'
                    }
                    x = requests.post(url,data = json.dumps(myobj),headers={"Content-Type": "application/json"})
                    
                    print(x.json())

                st.write(token_cont)
            warning(user_input)
        #################################################################################################################################################
        st.text("")
        st.text("")
        st.text("")
        st.subheader("Searching for Number of People Visited a Place...")
        Name2 = ""
        user_input2 = st.text_input("Enter a Location Name:", Name2)
        if user_input2:
            circles_size, people_active = bipartite.degrees(B,people1)

            def people_in_place(place):
                st.write('Number of People Visited the People: ', circles_size[place])

            people_in_place(user_input2)
        #################################################################################################################################################




header = st.container()

with header:
    st.title('Health Authority Dashboard!')
    st.text('This dashboard is for the usage of Health Authorities.')
    cs_body()







    







    



    
