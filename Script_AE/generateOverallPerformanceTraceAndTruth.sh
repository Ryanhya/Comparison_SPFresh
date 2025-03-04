setname="6c VectorPath=/home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/batches/base_embeddings"
truthname="18c TruthPath=/home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/batches/base_embeddings_truth"
querysize="12c QuerySize="

#update base vectors and query vectors
for i in {0..1000..100}
do
    
{  
    newsetname=$setname$i 
    newtruthname=$truthname$i 
    newquerysize=$querysize$[i+1] 
    sed -i "$newsetname" /home/lyh/SPFresh/Script_AE/iniFile/genTruth.ini  #修改genTruth.ini 文件中第六行的VectorPath值
    sed -i "$newtruthname" /home/lyh/SPFresh/Script_AE/iniFile/genTruth.ini  #修改genTruth.ini 文件中第十八行的TruthPath值
    sed -i "$newquerysize" /home/lyh/SPFresh/Script_AE/iniFile/genTruth.ini  #修改genTruth.ini 文件中第十二行的TruthPath值

    /home/lyh/SPFresh/Release/usefultool -BuildVector true --vectortype float --BaseVectorPath /home/lyh/datasets/embeddings/test/base_embeddings.bin --CurrentListFileName /home/lyh/datasets/embeddings/test/batches/base_embedding_ids --QueryVectorPath /home/lyh/datasets/embeddings/test/query_embeddings.bin --BaseVectorSplitPath /home/lyh/datasets/embeddings/test/query_vector_range.bin --filetype DEFAULT --ride 100 -d 256 -f DEFAULT --Batch $i -NewDataSetFileName /home/lyh/datasets/embeddings/test/batches/base_embeddings -NewQueryDataSetFileName /home/lyh/datasets/embeddings/test/batches/query_embeddings -ConfigurePath /home/lyh/SPFresh/Script_AE/iniFile/genTruth.ini
    
    
    
    #/home/lyh/SPFresh/Release/ssdserving /home/lyh/SPFresh/Script_AE/iniFile/genTruth.ini 
    
    
    
    #/home/lyh/SPFresh/Release/usefultool -ConvertTruth true --vectortype int8 --filetype DEFAULT --CurrentListFileName /home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/batches/base_embedding_ids -d 256 --Batch $i -f DEFAULT --truthPath /home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/truth/truth_embeddings_ --truthType DEFAULT --querySize $[i+1] --resultNum 10  
    
    
    /home/lyh/SPFresh/Release/usefultool -CallRecall true -resultNum 10 -queryPath /home/lyh/datasets/embeddings/test/batches/query_embeddings$i -searchResult /home/lyh/datasets/embeddings/test/batches/base_embeddings_truth$i.dist.bin -truthType DEFAULT -truthPath /home/lyh/datasets/embeddings/test/truth/truth_embeddings_$i -VectorPath /home/lyh/datasets/embeddings/test/batches/base_embeddings$i --vectortype float -d 256 -f DEFAULT |tee /home/lyh/datasets/embeddings/test/logs/log_spfresh_$i #if merge is needed, add -CallRecall true -resultNum 10 -searchResult /home/lyh/datasets/embeddings/test/batches/base_embeddings_truth$i.dist.bin -truthType DEFAULT -truthPath /home/lyh/datasets/embeddings/test/truth/truth_embeddings_$i.    other parameters already exists.
} 

done

rm -rf /home/lyh/datasets/embeddings/test/last_base_vector_id.txt
#get search results
#for i in {0..0..100}
#do

#{
    
#    /home/lyh/SPFresh/Release/ssdserving /home/lyh/SPFresh/Script_AE/iniFile/genTruth.ini 
    
    
    
#    /home/lyh/SPFresh/Release/usefultool -ConvertTruth true --vectortype int8 --filetype DEFAULT --CurrentListFileName /home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/batches/base_embedding_ids -d 256 --Batch $i -f DEFAULT --truthPath /home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/truth/truth_embeddings_ --truthType DEFAULT --querySize $[i+1] --resultNum 10  
    
    
#    /home/lyh/SPFresh/Release/usefultool -CallRecall true -resultNum 10 -queryPath /home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/batches/query_embeddings$i -searchResult /home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/batches/base_embeddings_truth$i.dist.bin -truthType DEFAULT -truthPath /home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/truth/truth_embeddings_$i.after -VectorPath /home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/batches/base_embeddings$i --vectortype int8 -d 256 -f DEFAULT |tee /home/lyh/datasets/argoverse2/motion_forecast/embeddings/test/logs/log_spfresh_$i 
    #searchResult对应的文件为ssdserving产生的结果，truthPath为usefultool的convertTruth选项产生的结果，其中convertTruth里的truthPath为groundTruth集的文件路径，而不是由ssdserving产生的
    #24984
#}&

#done

wait