import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import base64

# Function to perform matching and generate similarity scores
def perform_matching(origin_df, destination_df, selected_columns):
    # Combine the selected columns into a single text column for vectorization
    origin_df['combined_text'] = origin_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
    destination_df['combined_text'] = destination_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Use a pre-trained model for embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Vectorize the combined text
    origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
    destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

    # Create a FAISS index
    dimension = origin_embeddings.shape[1]  # The dimension of vectors
    faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
    faiss_index.add(destination_embeddings.astype('float32'))  # Add destination vectors to the index

    # Perform the search for the nearest neighbors
    D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)  # k=1 finds the closest match

    # Calculate similarity score (1 - normalized distance)
    similarity_scores = 1 - (D / np.max(D))

    # Create the output DataFrame with similarity score instead of distance
    matches_df = pd.DataFrame({
        'origin_url': origin_df['Address'],
        'matched_url': destination_df['Address'].iloc[I.flatten()].values,
        'similarity_score': np.round(similarity_scores.flatten(), 4)  # Rounded for better readability
    })

    return matches_df

def main():
    st.title('URL Redirect Similarity Matching App')

    origin_file = st.file_uploader("Upload origin.csv", type=['csv'], help="Please upload the CSV file containing the origin URLs", key="origin")
    destination_file = st.file_uploader("Upload destination.csv", type=['csv'], help="Please upload the CSV file containing the destination URLs", key="destination")

    if origin_file is not None and destination_file is not None:
        origin_df = pd.read_csv(origin_file)
        destination_df = pd.read_csv(destination_file)

        common_columns = list(set(origin_df.columns) & set(destination_df.columns))

        selected_columns = st.multiselect('Select the columns you want to include for similarity matching:', common_columns)

        if not selected_columns:
            st.warning("Please select at least one column to continue.")
        else:
            if st.button("Let's Go!"):
                matches_df = perform_matching(origin_df, destination_df, selected_columns)
                
                # Display matches in the app
                st.write(matches_df)

                # Save matches to a CSV file and provide download link
                csv = matches_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

    st.markdown("""
    ## Instructions
    
    This app performs similarity matching between two sets of URLs for the purpose of URL redirection mapping. 
    Please follow these instructions:

    **Step 1: Crawl your live website with Screaming Frog**
    
    You’ll need to perform a standard crawl on your website. Depending on how your website is built, this may or may not require a JavaScript crawl. The goal is to produce a list of as many accessible pages on your site as possible.
    
    **Step 2: Export HTML pages with 200 Status Code**
    
    Once the crawl has been completed, we want to export all of the found HTML URLs with a 200 Status Code. This will provide you with a list of our current live URLs and all of the default metadata Screaming Frog collects about them, such as Titles and Header Tags. Save this file as origin.csv.
    
    **Step 3: Repeat steps 1 and 2 for your staging website**
    
    We now need to gather the same data from our staging website, so we have something to compare to. Depending on how your staging site is secured, you may need to use features such as Screaming Frog’s forms authentication if password protected. Once the crawl has completed, you should export the data and save this file as destination.csv.
    
    **Optional: Find and replace your staging site domain or subdomain to match your live site**
    
    It’s likely your staging website is either on a different subdomain, TLD or even domain that won’t match our actual destination URL. For this reason, I will use a Find and Replace function on my destination.csv to change the path to match the final live site subdomain, domain or TLD.
    
    For example:
    
    - The live website is https://examplesite.com/ (origin.csv)
    - The staging website is https://test.examplesite.dev/ (destination.csv)
    
    The site is staying on the same domain; it’s just a redesign with different URLs, so I would open origin.csv and find any instance of https://examplesite.com/ and replace it with https://test.examplesite.dev/
    
    Find and Replace in Excel
    
    This also means when the redirect map is produced, the output is correct and only the final redirect logic needs to be written.
    """)

if __name__ == "__main__":
    main()
