#from email import header
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
st.set_page_config(layout="wide")
session = requests.Session()

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

xdata = fetch(session, f"https://jom-trace-backend.herokuapp.com/")
#

def cs_body():
    col1, col2 = st.columns(2)
    col1.title('One Mode Analysis')
    col2.title('Two Mode Analysis')

    df = pd.read_csv('social.csv')
    data= df.drop(['Location'], axis = 1)

    with col1:
        st.subheader("Social Network Graph")
        graph = nx.from_pandas_edgelist(data,source="From", target="To")
        fig = nx.draw_kamada_kawai(graph, with_labels=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        #################################################################################################################################################
        st.text("")
        st.text("")
        st.text("")
        st.subheader("Apriori Analysis")
        data_list = []
        for i in range (0,21):
            data_list.append([str(data.values[i,j]) for j in range(0,2)])
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
        
        country_map = {country:i for i, country in enumerate(data.stack().unique())}
        new_data=data.copy()

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
        st.subheader("Social Network Graph with Geography")
        people1 = list(df['From'])
        people2 = list(df['To'])
        place = list(df['Location'])

        def create_from_edgelist(top,bottom,middle):
            B = nx.Graph()
            for i in range(len(top)):
                B.add_node(top[i],bipartite=0)
                B.add_node(bottom[i],bipartite=1)
                B.add_node(middle[i],bipartite=2)
                B.add_edge(top[i],bottom[i])
                B.add_edge(bottom[i],middle[i])
            return B
    
        B = create_from_edgelist(place,people1,people2)

        f = plt.figure(1,figsize=(8,6),dpi=400)
        pos = nx.spring_layout(B)
        colors = {0:'r',1:'y',2:'y'}
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
            for i in range(len(people1)):
                ple.append([people1[i], place[i]])

            print(ple)

            ple2 = []
            for i in range(len(people1)):
                ple2.append([people1[i], people2[i]])

            people_cont = []
            place_cont = []
            warn = []
            no_warn = []
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

                output_df = pd.DataFrame({'People Close Contact' : [warn],
                                'Not Close Contact' : [no_warn]}, columns = ['People Close Contact', 'Not Close Contact'])
                st.write(output_df)
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
dataset = st.container()

page = st.sidebar.selectbox("Choose your page", ["Main Page", "One-Mode", "Two Mode"]) 

if page == "Main Page":
    df = pd.read_csv('social.csv')
    data= df.drop(['Location'], axis = 1)
    with header:
        st.title('Health Authority Dashboard!')
        st.text('This dashboard is for the usage of Health Authorities.')


    with dataset:
        data = pd.read_csv('social.csv')
        data_show = st.sidebar.checkbox("Show Dataset")
        if data_show:
            st.write(data)

    cs_body()









    







    



    
