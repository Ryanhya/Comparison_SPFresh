// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/VectorSetReaders/MemoryReader.h"
#include "inc/Core/SPANN/ExtraStaticSearcher.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include <shared_mutex>
#include <chrono>
#include <random>

#pragma warning(disable:4242)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4244)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4127)  // conditional expression is constant

namespace SPTAG
{
    namespace SPANN
    {
        std::atomic_int ExtraWorkSpace::g_spaceCount(0);
        EdgeCompare Selection::g_edgeComparer;
        template <typename T>
        thread_local std::shared_ptr<ExtraWorkSpace> Index<T>::m_workspace;

        std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO = []() -> std::shared_ptr<Helper::DiskIO> { return std::shared_ptr<Helper::DiskIO>(new Helper::AsyncFileIO()); };

        template <typename T>
        bool Index<T>::CheckHeadIndexType() {
            SPTAG::VectorValueType v1 = m_index->GetVectorValueType(), v2 = GetEnumValueType<T>();
            if (v1 != v2) {
                LOG(Helper::LogLevel::LL_Error, "Head index and vectors don't have the same value types, which are %s %s\n",
                    SPTAG::Helper::Convert::ConvertToString(v1).c_str(),
                    SPTAG::Helper::Convert::ConvertToString(v2).c_str()
                );
                if (!m_pQuantizer) return false;
            }
            return true;
        }

        template <typename T>
        void Index<T>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
        {
            m_pQuantizer = quantizer;
            if (m_pQuantizer)
            {
                m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase() : 1;
            }
            else
            {
                m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<std::uint8_t>() * COMMON::Utils::GetBase<std::uint8_t>() : 1;
            }
            if (m_index)
            {
                m_index->SetQuantizer(quantizer);
            }
        }

        template <typename T>
        ErrorCode Index<T>::LoadConfig(Helper::IniReader& p_reader)
        {
            IndexAlgoType algoType = p_reader.GetParameter("Base", "IndexAlgoType", IndexAlgoType::Undefined);
            VectorValueType valueType = p_reader.GetParameter("Base", "ValueType", VectorValueType::Undefined);
            if ((m_index = CreateInstance(algoType, valueType)) == nullptr) return ErrorCode::FailedParseValue;

            std::string sections[] = { "Base", "SelectHead", "BuildHead", "BuildSSDIndex" };
            for (int i = 0; i < 4; i++) {
                auto parameters = p_reader.GetParameters(sections[i].c_str());
                for (auto iter = parameters.begin(); iter != parameters.end(); iter++) {
                    SetParameter(iter->first.c_str(), iter->second.c_str(), sections[i].c_str());
                }
            }

            if (m_pQuantizer)
            {
                m_pQuantizer->SetEnableADC(m_options.m_enableADC);
            }

            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs)
        {
            m_index->SetQuantizer(m_pQuantizer);
            if (m_index->LoadIndexDataFromMemory(p_indexBlobs) != ErrorCode::Success) return ErrorCode::Fail;

            m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
            //m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
            //m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
            m_index->UpdateIndex();
            m_index->SetReady(true);

            if (m_pQuantizer)
            {
                m_extraSearcher.reset(new ExtraStaticSearcher<std::uint8_t>());
            }
            else
            {
                if (m_options.m_useKV) {
                    if (m_options.m_inPlace) {
                        m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options.m_KVPath.c_str(), m_options.m_dim, INT_MAX, m_options.m_useDirectIO, m_options.m_latencyLimit, m_options.m_mergeThreshold));
                    }
                    else {
                        m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options.m_KVPath.c_str(), m_options.m_dim, m_options.m_postingPageLimit * PageSize / (sizeof(T) * m_options.m_dim + sizeof(int) + sizeof(uint8_t)), m_options.m_useDirectIO, m_options.m_latencyLimit, m_options.m_mergeThreshold));
                    }
                }
                else {
                    m_extraSearcher.reset(new ExtraStaticSearcher<T>());
                }
            }

            if (!m_extraSearcher->LoadIndex(m_options, m_versionMap)) return ErrorCode::Fail;

            if (m_options.m_excludehead) m_vectorTranslateMap.reset((std::uint64_t*)(p_indexBlobs.back().Data()), [=](std::uint64_t* ptr) {});

            omp_set_num_threads(m_options.m_iSSDNumberOfThreads);
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams)
        {
            m_index->SetQuantizer(m_pQuantizer);
            if (m_index->LoadIndexData(p_indexStreams) != ErrorCode::Success) return ErrorCode::Fail;

            m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
            m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
            m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
            m_index->UpdateIndex();
            m_index->SetReady(true);

            // TODO: Choose an extra searcher based on config
            // Not Ready
            if (m_pQuantizer)
            {
                m_extraSearcher.reset(new ExtraStaticSearcher<std::uint8_t>());
            }
            else
            {
                if (m_options.m_useKV) {
                    if (m_options.m_inPlace) {
                        m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options.m_KVPath.c_str(), m_options.m_dim, INT_MAX, m_options.m_useDirectIO, m_options.m_latencyLimit, m_options.m_mergeThreshold));
                    }
                    else {
                        m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options.m_KVPath.c_str(), m_options.m_dim, m_options.m_postingPageLimit  * PageSize / (sizeof(T) * m_options.m_dim + sizeof(int) + sizeof(uint8_t)), m_options.m_useDirectIO, m_options.m_latencyLimit, m_options.m_mergeThreshold));
                    }
                }
                else if (m_options.m_useSPDK) {
                    m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options.m_spdkMappingPath.c_str(), m_options.m_dim, m_options.m_postingPageLimit, m_options.m_useDirectIO, m_options.m_latencyLimit, m_options.m_mergeThreshold, true, m_options.m_spdkBatchSize, m_options.m_bufferLength));
                } else {
                    m_extraSearcher.reset(new ExtraStaticSearcher<T>());
                }
            }

            if (!m_extraSearcher->LoadIndex(m_options, m_versionMap)) return ErrorCode::Fail;

            if (m_options.m_excludehead) {
                m_vectorTranslateMap.reset(new std::uint64_t[m_index->GetNumSamples()], std::default_delete<std::uint64_t[]>());
                IOBINARY(p_indexStreams[m_index->GetIndexFiles()->size()], ReadBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), reinterpret_cast<char*>(m_vectorTranslateMap.get()));
            }

            omp_set_num_threads(m_options.m_iSSDNumberOfThreads);

            if (m_options.m_useSPDK) {
                int m_vectorLimit = m_options.m_postingPageLimit * PageSize / (sizeof(T) * m_options.m_dim + sizeof(int) + sizeof(uint8_t));
                m_versionMap.Initialize(m_options.m_vectorSize, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
                int m_vectorInfoSize = sizeof(T) * m_options.m_dim + sizeof(int) + sizeof(uint8_t);
                LOG(Helper::LogLevel::LL_Info, "Copying data from static to SPDK\n");
                std::shared_ptr<IExtraSearcher> storeExtraSearcher;
                storeExtraSearcher.reset(new ExtraStaticSearcher<T>());
                if (!storeExtraSearcher->LoadIndex(m_options, m_versionMap)) {
                    LOG(Helper::LogLevel::LL_Info, "Initialize Error\n");
                    exit(1);
                }
                int totalPostingNum = m_index->GetNumSamples();
                m_extraSearcher->InitPostingRecord(m_index);

                std::vector<std::thread> threads;
                std::atomic_size_t vectorsSent(0);

                auto func = [&]()
                {
                    m_extraSearcher->Initialize();
                    size_t index = 0;
                    while (true)
                    {
                        index = vectorsSent.fetch_add(1);
                        if (index < totalPostingNum)
                        {

                            if ((index & ((1 << 14) - 1)) == 0)
                            {
                                LOG(Helper::LogLevel::LL_Info, "Copy to SPDK: Sent %.2lf%%...\n", index * 100.0 / totalPostingNum);
                            }
                            std::string tempPosting;
                            storeExtraSearcher->GetWritePosting(index, tempPosting);
                            int vectorNum = (int)(tempPosting.size() / (m_vectorInfoSize - sizeof(uint8_t)));

                            if (vectorNum > m_vectorLimit) vectorNum = m_vectorLimit;

                            auto* postingP = reinterpret_cast<char*>(&tempPosting.front());
                            std::string newPosting(m_vectorInfoSize * vectorNum , '\0');
                            char* ptr = (char*)(newPosting.c_str());
                            for (int j = 0; j < vectorNum; ++j, ptr += m_vectorInfoSize) {
                                char* vectorInfo = postingP + j * (m_vectorInfoSize - sizeof(uint8_t));
                                int VID = *(reinterpret_cast<int*>(vectorInfo));
                                uint8_t version = m_versionMap.GetVersion(VID);
                                memcpy(ptr, &VID, sizeof(int));
                                memcpy(ptr + sizeof(int), &version, sizeof(uint8_t));
                                memcpy(ptr + sizeof(int) + sizeof(uint8_t), vectorInfo + sizeof(int), m_vectorInfoSize - sizeof(uint8_t) - sizeof(int));
                            }

                            if (m_options.m_excludehead) {
                                auto VIDTrans = static_cast<SizeType>((m_vectorTranslateMap.get())[index]);
                                uint8_t version = m_versionMap.GetVersion(VIDTrans);
                                std::string appendPosting(m_vectorInfoSize, '\0');
                                char* ptr = (char*)(appendPosting.c_str());
                                memcpy(ptr, &VIDTrans, sizeof(VIDTrans));
                                memcpy(ptr + sizeof(VIDTrans), &version, sizeof(version));
                                memcpy(ptr + sizeof(int) + sizeof(uint8_t), m_index->GetSample(index), m_vectorInfoSize - sizeof(int) + sizeof(uint8_t));
                                newPosting = appendPosting + newPosting;
                            }

                            m_extraSearcher->GetWritePosting(index, newPosting, true);
                        }
                        else
                        {
                            m_extraSearcher->ExitBlockController();
                            return;
                        }
                    }
                };
            for (int j = 0; j < m_options.m_iSSDNumberOfThreads; j++) { threads.emplace_back(func); }
            for (auto& thread : threads) { thread.join(); }
            } else {
                m_versionMap.Load(m_options.m_deleteIDFile, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
            }

            if ((m_options.m_useSPDK || m_options.m_useKV) && m_options.m_preReassign) {
                std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(m_options.m_valueType, m_options.m_dim, m_options.m_vectorType, m_options.m_vectorDelimiter, m_options.m_iSSDNumberOfThreads));
                auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (m_options.m_vectorPath.empty())
                {
                    LOG(Helper::LogLevel::LL_Info, "Vector file is empty. Skipping loading.\n");
                }
                else {
                    if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_vectorPath))
                    {
                        LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
                        return ErrorCode::Fail;
                    }
                    // m_options.m_vectorSize = vectorReader->GetVectorSet()->Count();
                }
                m_extraSearcher->RefineIndex(vectorReader, m_index);
            }

            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::SaveConfig(std::shared_ptr<Helper::DiskIO> p_configOut)
        {
            IOSTRING(p_configOut, WriteString, "[Base]\n");
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr) \
                IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str()); \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBasicParameter

            IOSTRING(p_configOut, WriteString, "[SelectHead]\n");
#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
                IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str()); \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

            IOSTRING(p_configOut, WriteString, "[BuildHead]\n");
#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
                IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str()); \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

            m_index->SaveConfig(p_configOut);

            Helper::Convert::ConvertStringTo<int>(m_index->GetParameter("HashTableExponent").c_str(), m_options.m_hashExp);
            IOSTRING(p_configOut, WriteString, "[BuildSSDIndex]\n");
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr) \
                IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str()); \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter

            IOSTRING(p_configOut, WriteString, "\n");
            return ErrorCode::Success;
        }

        template<typename T>
        ErrorCode Index<T>::SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams)
        {
            if (m_index == nullptr) return ErrorCode::EmptyIndex;

            ErrorCode ret;
            if ((ret = m_index->SaveIndexData(p_indexStreams)) != ErrorCode::Success) return ret;

            if (m_options.m_excludehead) IOBINARY(p_indexStreams[m_index->GetIndexFiles()->size()], WriteBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), (char*)(m_vectorTranslateMap.get()));
            m_versionMap.Save(m_options.m_deleteIDFile);
            return ErrorCode::Success;
        }

#pragma region K-NN search

        template<typename T>
        ErrorCode Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
        {
            if (!m_bReady) return ErrorCode::EmptyIndex;

            COMMON::QueryResultSet<T>* p_queryResults;
            if (p_query.GetResultNum() >= m_options.m_searchInternalResultNum)
                p_queryResults = (COMMON::QueryResultSet<T>*) & p_query;
            else
                p_queryResults = new COMMON::QueryResultSet<T>((const T*)p_query.GetTarget(), m_options.m_searchInternalResultNum);

            m_index->SearchIndex(*p_queryResults);

            if (m_extraSearcher != nullptr) {
                if (m_workspace.get() == nullptr) {
                    m_workspace.reset(new ExtraWorkSpace());
                    m_workspace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, min(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
                }
                m_workspace->m_deduper.clear();
                m_workspace->m_postingIDs.clear();

                float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
                for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
                {
                    auto res = p_queryResults->GetResult(i);
                    if (res->VID == -1) break;
                    
                    auto postingID = res->VID;
                    if (m_vectorTranslateMap.get() != nullptr) res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                    else {
                        res->VID = -1;
                        res->Dist = MaxDist;
                    }

                    // Don't do disk reads for irrelevant pages
                    if (m_workspace->m_postingIDs.size() >= m_options.m_searchInternalResultNum ||
                        (limitDist > 0.1 && res->Dist > limitDist) ||
                        !m_extraSearcher->CheckValidPosting(postingID))
                        continue;
                    m_workspace->m_postingIDs.emplace_back(postingID);
                }

                if (m_vectorTranslateMap.get() != nullptr) p_queryResults->Reverse();
                m_extraSearcher->SearchIndex(m_workspace.get(), *p_queryResults, m_index, nullptr);
                p_queryResults->SortResult();
            }

            if (p_query.GetResultNum() < m_options.m_searchInternalResultNum) {
                std::copy(p_queryResults->GetResults(), p_queryResults->GetResults() + p_query.GetResultNum(), p_query.GetResults());
                delete p_queryResults;
            }

            if (p_query.WithMeta() && nullptr != m_pMetadata)
            {
                for (int i = 0; i < p_query.GetResultNum(); ++i)
                {
                    SizeType result = p_query.GetResult(i)->VID;
                    p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
                }
            }
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::SearchDiskIndex(QueryResult& p_query, SearchStats* p_stats) const
        {
            if (nullptr == m_extraSearcher) return ErrorCode::EmptyIndex;

            COMMON::QueryResultSet<T>* p_queryResults = (COMMON::QueryResultSet<T>*) & p_query;

            if (m_workspace.get() == nullptr) {
                m_workspace.reset(new ExtraWorkSpace());
                m_workspace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, min(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            m_workspace->m_deduper.clear();
            m_workspace->m_postingIDs.clear();

            float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
            int i = 0;
            for (; i < p_queryResults->GetResultNum(); ++i)
            {
                auto res = p_queryResults->GetResult(i);
                if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist)) break;
                if (m_extraSearcher->CheckValidPosting(res->VID))
                {
                    m_workspace->m_postingIDs.emplace_back(res->VID);
                }
                if (m_vectorTranslateMap.get() != nullptr) res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                else {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }

            for (; i < p_queryResults->GetResultNum(); ++i)
            {
                auto res = p_queryResults->GetResult(i);
                if (res->VID == -1) break;
                if (m_vectorTranslateMap.get() != nullptr)  res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                else {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }
            if (m_vectorTranslateMap.get() != nullptr) p_queryResults->Reverse();
            m_extraSearcher->SearchIndex(m_workspace.get(), *p_queryResults, m_index, p_stats);
            p_queryResults->SortResult();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::DebugSearchDiskIndex(QueryResult& p_query, int p_subInternalResultNum, int p_internalResultNum,
            SearchStats* p_stats, std::set<int>* truth, std::map<int, std::set<int>>* found) const
        {
            if (nullptr == m_extraSearcher) return ErrorCode::EmptyIndex;

            std::unique_ptr<COMMON::QueryResultSet<T>> newResults;
            if (m_vectorTranslateMap.get() != nullptr) {
                newResults.reset(new COMMON::QueryResultSet<T>(*((COMMON::QueryResultSet<T>*) & p_query)));
                for (int i = 0; i < newResults->GetResultNum(); ++i)
                {
                    auto res = newResults->GetResult(i);
                    if (res->VID == -1) break;

                    auto global_VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                    if (truth && truth->count(global_VID)) (*found)[res->VID].insert(global_VID);
                    res->VID = global_VID;
                }
                newResults->Reverse();
            }
            else {
                newResults.reset(new COMMON::QueryResultSet<T>((T*)p_query.GetTarget(), p_query.GetResultNum()));
            }

            if (m_workspace.get() == nullptr) {
                m_workspace.reset(new ExtraWorkSpace());
                m_workspace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, min(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            m_workspace->m_deduper.clear();

            int partitions = (p_internalResultNum + p_subInternalResultNum - 1) / p_subInternalResultNum;
            float limitDist = p_query.GetResult(0)->Dist * m_options.m_maxDistRatio;
            for (SizeType p = 0; p < partitions; p++) {
                int subInternalResultNum = min(p_subInternalResultNum, p_internalResultNum - p_subInternalResultNum * p);

                m_workspace->m_postingIDs.clear();

                for (int i = p * p_subInternalResultNum; i < p * p_subInternalResultNum + subInternalResultNum; i++)
                {
                    auto res = p_query.GetResult(i);
                    if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist)) break;
                    if (!m_extraSearcher->CheckValidPosting(res->VID)) continue;
                    m_workspace->m_postingIDs.emplace_back(res->VID);
                }

                m_extraSearcher->SearchIndex(m_workspace.get(), *newResults, m_index, p_stats, truth, found);
            }

            newResults->SortResult();
            std::copy(newResults->GetResults(), newResults->GetResults() + newResults->GetResultNum(), p_query.GetResults());

            return ErrorCode::Success;
        }
#pragma endregion

        template <typename T>
        void Index<T>::SelectHeadAdjustOptions(int p_vectorCount) {
            LOG(Helper::LogLevel::LL_Info, "Begin Adjust Parameters...\n");

            if (m_options.m_headVectorCount != 0) m_options.m_ratio = m_options.m_headVectorCount * 1.0 / p_vectorCount;
            int headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
            if (headCnt == 0)
            {
                for (double minCnt = 1; headCnt == 0; minCnt += 0.2)
                {
                    m_options.m_ratio = minCnt / p_vectorCount;
                    headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
                }

                LOG(Helper::LogLevel::LL_Info, "Setting requires to select none vectors as head, adjusted it to %d vectors\n", headCnt);
            }

            if (m_options.m_iBKTKmeansK > headCnt)
            {
                m_options.m_iBKTKmeansK = headCnt;
                LOG(Helper::LogLevel::LL_Info, "Setting of cluster number is less than head count, adjust it to %d\n", headCnt);
            }

            if (m_options.m_selectThreshold == 0)
            {
                m_options.m_selectThreshold = min(p_vectorCount - 1, static_cast<int>(1 / m_options.m_ratio));
                LOG(Helper::LogLevel::LL_Info, "Set SelectThreshold to %d\n", m_options.m_selectThreshold);
            }

            if (m_options.m_splitThreshold == 0)
            {
                m_options.m_splitThreshold = min(p_vectorCount - 1, static_cast<int>(m_options.m_selectThreshold * 2));
                LOG(Helper::LogLevel::LL_Info, "Set SplitThreshold to %d\n", m_options.m_splitThreshold);
            }

            if (m_options.m_splitFactor == 0)
            {
                m_options.m_splitFactor = min(p_vectorCount - 1, static_cast<int>(std::round(1 / m_options.m_ratio) + 0.5));
                LOG(Helper::LogLevel::LL_Info, "Set SplitFactor to %d\n", m_options.m_splitFactor);
            }
        }

        template <typename T>
        int Index<T>::SelectHeadDynamicallyInternal(const std::shared_ptr<COMMON::BKTree> p_tree, int p_nodeID,
            const Options& p_opts, std::vector<int>& p_selected)
        {
            typedef std::pair<int, int> CSPair;
            std::vector<CSPair> children;
            int childrenSize = 1;
            const auto& node = (*p_tree)[p_nodeID];
            if (node.childStart >= 0)
            {
                children.reserve(node.childEnd - node.childStart);
                for (int i = node.childStart; i < node.childEnd; ++i)
                {
                    int cs = SelectHeadDynamicallyInternal(p_tree, i, p_opts, p_selected);
                    if (cs > 0)
                    {
                        children.emplace_back(i, cs);
                        childrenSize += cs;
                    }
                }
            }

            if (childrenSize >= p_opts.m_selectThreshold)
            {
                if (node.centerid < (*p_tree)[0].centerid)
                {
                    p_selected.push_back(node.centerid);
                }

                if (childrenSize > p_opts.m_splitThreshold)
                {
                    std::sort(children.begin(), children.end(), [](const CSPair& a, const CSPair& b)
                    {
                        return a.second > b.second;
                    });

                    size_t selectCnt = static_cast<size_t>(std::ceil(childrenSize * 1.0 / p_opts.m_splitFactor) + 0.5);
                    //if (selectCnt > 1) selectCnt -= 1;
                    for (size_t i = 0; i < selectCnt && i < children.size(); ++i)
                    {
                        p_selected.push_back((*p_tree)[children[i].first].centerid);
                    }
                }

                return 0;
            }

            return childrenSize;
        }

        template <typename T>
        void Index<T>::SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree, int p_vectorCount, std::vector<int>& p_selected) {
            p_selected.clear();
            p_selected.reserve(p_vectorCount);

            if (static_cast<int>(std::round(m_options.m_ratio * p_vectorCount)) >= p_vectorCount)
            {
                for (int i = 0; i < p_vectorCount; ++i)
                {
                    p_selected.push_back(i);
                }

                return;
            }
            Options opts = m_options;

            int selectThreshold = m_options.m_selectThreshold;
            int splitThreshold = m_options.m_splitThreshold;

            double minDiff = 100;
            for (int select = 2; select <= m_options.m_selectThreshold; ++select)
            {
                opts.m_selectThreshold = select;
                opts.m_splitThreshold = m_options.m_splitThreshold;

                int l = m_options.m_splitFactor;
                int r = m_options.m_splitThreshold;

                while (l < r - 1)
                {
                    opts.m_splitThreshold = (l + r) / 2;
                    p_selected.clear();

                    SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
                    std::sort(p_selected.begin(), p_selected.end());
                    p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());

                    double diff = static_cast<double>(p_selected.size()) / p_vectorCount - m_options.m_ratio;

                    LOG(Helper::LogLevel::LL_Info,
                        "Select Threshold: %d, Split Threshold: %d, diff: %.2lf%%.\n",
                        opts.m_selectThreshold,
                        opts.m_splitThreshold,
                        diff * 100.0);

                    if (minDiff > fabs(diff))
                    {
                        minDiff = fabs(diff);

                        selectThreshold = opts.m_selectThreshold;
                        splitThreshold = opts.m_splitThreshold;
                    }

                    if (diff > 0)
                    {
                        l = (l + r) / 2;
                    }
                    else
                    {
                        r = (l + r) / 2;
                    }
                }
            }

            opts.m_selectThreshold = selectThreshold;
            opts.m_splitThreshold = splitThreshold;

            LOG(Helper::LogLevel::LL_Info,
                "Final Select Threshold: %d, Split Threshold: %d.\n",
                opts.m_selectThreshold,
                opts.m_splitThreshold);

            p_selected.clear();
            SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
            std::sort(p_selected.begin(), p_selected.end());
            p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());
        }

        template <typename T>
        template <typename InternalDataType>
        bool Index<T>::SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader) {
            std::shared_ptr<VectorSet> vectorset = p_reader->GetVectorSet();
            if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized())
                vectorset->Normalize(m_options.m_iSelectHeadNumberOfThreads);

            LOG(Helper::LogLevel::LL_Info, "Begin initial data (%d,%d)...\n", vectorset->Count(), vectorset->Dimension());

            COMMON::Dataset<InternalDataType> data(vectorset->Count(), vectorset->Dimension(), vectorset->Count(), vectorset->Count() + 1, (InternalDataType*)vectorset->GetData());

            auto t1 = std::chrono::high_resolution_clock::now();
            SelectHeadAdjustOptions(data.R());
            std::vector<int> selected;
            if (data.R() == 1) {
                selected.push_back(0);
            }
            else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "Random")) {
                LOG(Helper::LogLevel::LL_Info, "Start generating Random head.\n");
                selected.resize(data.R());
                for (int i = 0; i < data.R(); i++) selected[i] = i;
                std::shuffle(selected.begin(), selected.end(), rg);
                int headCnt = static_cast<int>(std::round(m_options.m_ratio * data.R()));
                selected.resize(headCnt);
            }
            else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "BKT")) {
                LOG(Helper::LogLevel::LL_Info, "Start generating BKT.\n");
                std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
                bkt->m_iBKTKmeansK = m_options.m_iBKTKmeansK;
                bkt->m_iBKTLeafSize = m_options.m_iBKTLeafSize;
                bkt->m_iSamples = m_options.m_iSamples;
                bkt->m_iTreeNumber = m_options.m_iTreeNumber;
                bkt->m_fBalanceFactor = m_options.m_fBalanceFactor;
                LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
                LOG(Helper::LogLevel::LL_Info, "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, BKTLambdaFactor:%f TreeNumber: %d, ThreadNum: %d.\n",
                    bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_fBalanceFactor, bkt->m_iTreeNumber, m_options.m_iSelectHeadNumberOfThreads);

                bkt->BuildTrees<InternalDataType>(data, m_options.m_distCalcMethod, m_options.m_iSelectHeadNumberOfThreads, nullptr, nullptr, true);
                auto t2 = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
                LOG(Helper::LogLevel::LL_Info, "End invoking BuildTrees.\n");
                LOG(Helper::LogLevel::LL_Info, "Invoking BuildTrees used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);

                if (m_options.m_saveBKT) {
                    std::stringstream bktFileNameBuilder;
                    bktFileNameBuilder << m_options.m_vectorPath << ".bkt." << m_options.m_iBKTKmeansK << "_"
                                       << m_options.m_iBKTLeafSize << "_" << m_options.m_iTreeNumber << "_" << m_options.m_iSamples << "_"
                                       << static_cast<int>(m_options.m_distCalcMethod) << ".bin";
                    bkt->SaveTrees(bktFileNameBuilder.str());
                }
                LOG(Helper::LogLevel::LL_Info, "Finish generating BKT.\n");

                LOG(Helper::LogLevel::LL_Info, "Start selecting nodes...Select Head Dynamically...\n");
                SelectHeadDynamically(bkt, data.R(), selected);

                if (selected.empty()) {
                    LOG(Helper::LogLevel::LL_Error, "Can't select any vector as head with current settings\n");
                    return false;
                }
            }

            LOG(Helper::LogLevel::LL_Info,
                "Seleted Nodes: %u, about %.2lf%% of total.\n",
                static_cast<unsigned int>(selected.size()),
                selected.size() * 100.0 / data.R());

            if (!m_options.m_noOutput)
            {
                std::sort(selected.begin(), selected.end());

                std::shared_ptr<Helper::DiskIO> output = SPTAG::f_createIO(), outputIDs = SPTAG::f_createIO();
                if (output == nullptr || outputIDs == nullptr ||
                    !output->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(), std::ios::binary | std::ios::out) ||
                    !outputIDs->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::out)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to create output file:%s %s\n",
                        (m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(),
                        (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                    return false;
                }

                SizeType val = static_cast<SizeType>(selected.size());
                if (output->WriteBinary(sizeof(val), reinterpret_cast<char*>(&val)) != sizeof(val)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                    return false;
                }
                DimensionType dt = data.C();
                if (output->WriteBinary(sizeof(dt), reinterpret_cast<char*>(&dt)) != sizeof(dt)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                    return false;
                }

                for (int i = 0; i < selected.size(); i++)
                {
                    uint64_t vid = static_cast<uint64_t>(selected[i]);
                    if (outputIDs->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                        return false;
                    }

                    if (output->WriteBinary(sizeof(InternalDataType) * data.C(), (char*)(data[vid])) != sizeof(InternalDataType) * data.C()) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                        return false;
                    }
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t3 - t1).count();
            LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
            return true;
        }

        template <typename T>
        ErrorCode Index<T>::BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader) {
            if (!m_options.m_indexDirectory.empty()) {
                if (!direxists(m_options.m_indexDirectory.c_str()))
                {
                    mkdir(m_options.m_indexDirectory.c_str());
                }
            }

            LOG(Helper::LogLevel::LL_Info, "Begin Select Head...\n");
            auto t1 = std::chrono::high_resolution_clock::now();
            if (m_options.m_selectHead) {
                omp_set_num_threads(m_options.m_iSelectHeadNumberOfThreads);//SelectHead->NumberOfThreads
                bool success = false;
                if (m_pQuantizer)
                {
                    success = SelectHeadInternal<std::uint8_t>(p_reader);
                }
                else
                {
                    success = SelectHeadInternal<T>(p_reader);
                }
                if (!success) {
                    LOG(Helper::LogLevel::LL_Error, "SelectHead Failed!\n");
                    return ErrorCode::Fail;
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            double selectHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
            LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs\n", selectHeadTime);

            LOG(Helper::LogLevel::LL_Info, "Begin Build Head...\n");
            if (m_options.m_buildHead) {
                auto valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
                auto dims = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;

                m_index = SPTAG::VectorIndex::CreateInstance(m_options.m_indexAlgoType, valueType);
                m_index->SetParameter("DistCalcMethod", SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
                m_index->SetQuantizer(m_pQuantizer);
                for (const auto& iter : m_headParameters)
                {
                    m_index->SetParameter(iter.first.c_str(), iter.second.c_str());
                }

                std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(valueType, dims, VectorFileType::DEFAULT));
                auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read head vector file.\n");
                    return ErrorCode::Fail;
                }
                {
                    auto headvectorset = vectorReader->GetVectorSet();
                    if (m_index->BuildIndex(headvectorset, nullptr, false, true, true) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to build head index.\n");
                        return ErrorCode::Fail;
                    }
                    m_index->SetQuantizerFileName(m_options.m_quantizerFilePath.substr(m_options.m_quantizerFilePath.find_last_of("/\\") + 1));
                    if (m_index->SaveIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to save head index.\n");
                        return ErrorCode::Fail;
                    }
                }
                m_index.reset();
                if (LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_index) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            double buildHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count();
            LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs\n", selectHeadTime, buildHeadTime);
            LOG(Helper::LogLevel::LL_Info, "Begin Build SSDIndex...\n");
            if (m_options.m_enableSSD) {
                omp_set_num_threads(m_options.m_iSSDNumberOfThreads);//BuildSSDIndex->NumberOfThreads

                if (m_index == nullptr && LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_index) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
                    return ErrorCode::Fail;
                }
                m_index->SetQuantizer(m_pQuantizer);
                if (!CheckHeadIndexType()) return ErrorCode::Fail;

                m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
                m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
                m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
                m_index->UpdateIndex();

                if (m_options.m_useKV)
                {
                    if (m_options.m_inPlace) {
                        m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options.m_KVPath.c_str(), m_options.m_dim, INT_MAX, m_options.m_useDirectIO, m_options.m_latencyLimit, m_options.m_mergeThreshold));
                    }
                    else {
                        m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options.m_KVPath.c_str(), m_options.m_dim, m_options.m_postingPageLimit  * PageSize / (sizeof(T)*m_options.m_dim + sizeof(int) + sizeof(uint8_t)), m_options.m_useDirectIO, m_options.m_latencyLimit, m_options.m_mergeThreshold));
                    }
                } else if (m_options.m_useSPDK)
                {
                    if (m_options.m_inPlace) {
                        LOG(Helper::LogLevel::LL_Info, "Currently unsupport SPDK with inplace!\n");
                        exit(1);
                    }
                    else {
                        m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options.m_spdkMappingPath.c_str(), m_options.m_dim, m_options.m_postingPageLimit, m_options.m_useDirectIO, m_options.m_latencyLimit, m_options.m_mergeThreshold, true, m_options.m_spdkBatchSize));
                    }  
                }
                else {
                    if (m_pQuantizer) {
                        m_extraSearcher.reset(new ExtraStaticSearcher<std::uint8_t>());
                    }
                    else {
                        m_extraSearcher.reset(new ExtraStaticSearcher<T>());
                    }
                }

                if (m_options.m_buildSsdIndex) {
                    if (!m_options.m_excludehead) {
                        LOG(Helper::LogLevel::LL_Info, "Include all vectors into SSD index...\n");
                        if (fileexists((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str()) &&
                            remove((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str()) != 0) {
                            LOG(Helper::LogLevel::LL_Warning, "Head vector file can't be removed.\n");
                        }
                    }

                    if (!m_extraSearcher->BuildIndex(p_reader, m_index, m_options, m_versionMap)) {
                        LOG(Helper::LogLevel::LL_Error, "BuildSSDIndex Failed!\n");
                        if (m_options.m_buildSsdIndex) {
                            return ErrorCode::Fail;
                        }
                        else {
                            m_extraSearcher.reset();
                        }
                    }
                }
                if (!m_extraSearcher->LoadIndex(m_options, m_versionMap)) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot Load SSDIndex!\n");
                    return ErrorCode::Fail;
                }

                if (m_extraSearcher != nullptr) {
                    if (m_options.m_excludehead) {
                        m_vectorTranslateMap.reset(new std::uint64_t[m_index->GetNumSamples()], std::default_delete<std::uint64_t[]>());
                        std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
                        if (ptr == nullptr || !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::in)) {
                            LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                            return ErrorCode::Fail;
                        }
                        IOBINARY(ptr, ReadBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), (char*)(m_vectorTranslateMap.get()));
                    }
                    if ((m_options.m_useKV || m_options.m_useSPDK) && m_options.m_preReassign) {
                        m_extraSearcher->RefineIndex(p_reader, m_index);
                    }
                }
            }
            
            auto t4 = std::chrono::high_resolution_clock::now();
            double buildSSDTime = std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
            LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs build ssd time: %.2lfs\n", selectHeadTime, buildHeadTime, buildSSDTime);

            if (m_options.m_deleteHeadVectors) {
                if (fileexists((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) &&
                    remove((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) != 0) {
                    LOG(Helper::LogLevel::LL_Warning, "Head vector file can't be removed.\n");
                }
            }

            m_bReady = true;
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::BuildIndex(bool p_normalized)
        {
            SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
            SizeType dim = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
            std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(valueType, dim, m_options.m_vectorType, m_options.m_vectorDelimiter, m_options.m_iSSDNumberOfThreads, p_normalized));
            auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
            if (m_options.m_vectorPath.empty())
            {
                LOG(Helper::LogLevel::LL_Info, "Vector file is empty. Skipping loading.\n");
            }
            else {
                if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_vectorPath))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
                    return ErrorCode::Fail;
                }
                m_options.m_vectorSize = vectorReader->GetVectorSet()->Count();
            }

            return BuildIndexInternal(vectorReader);
        }

        template <typename T>
        ErrorCode Index<T>::BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized, bool p_shareOwnership)
        {
            if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0) return ErrorCode::EmptyData;

            std::shared_ptr<VectorSet> vectorSet;
            if (p_shareOwnership) {
                vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                    GetEnumValueType<T>(), p_dimension, p_vectorNum));
            }
            else {
                ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
                memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
                vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
            }


            if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized) {
                vectorSet->Normalize(m_options.m_iSSDNumberOfThreads);
            }
            SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
            std::shared_ptr<Helper::VectorSetReader> vectorReader(new Helper::MemoryVectorReader(std::make_shared<Helper::ReaderOptions>(valueType, p_dimension, VectorFileType::DEFAULT, m_options.m_vectorDelimiter, m_options.m_iSSDNumberOfThreads, true),
                vectorSet));

            m_options.m_valueType = GetEnumValueType<T>();
            m_options.m_dim = p_dimension;
            m_options.m_vectorSize = p_vectorNum;
            return BuildIndexInternal(vectorReader);
        }

        template <typename T>
        ErrorCode Index<T>::UpdateIndex()
        {
            omp_set_num_threads(m_options.m_iSSDNumberOfThreads);
            m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
            //m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
            //m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
            m_index->UpdateIndex();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::SetParameter(const char* p_param, const char* p_value, const char* p_section)
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") && !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute")) {
                if (m_index != nullptr) return m_index->SetParameter(p_param, p_value);
                else m_headParameters[p_param] = p_value;
            }
            else {
                m_options.SetParameter(p_section, p_param, p_value);
            }
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistCalcMethod")) {
                if (m_pQuantizer)
                {
                    m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                    m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase() : 1;
                }
                else
                {
                    m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                    m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
                }
            }
            return ErrorCode::Success;
        }

        template <typename T>
        std::string Index<T>::GetParameter(const char* p_param, const char* p_section) const
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") && !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute")) {
                if (m_index != nullptr) return m_index->GetParameter(p_param);
                else {
                    auto iter = m_headParameters.find(p_param);
                    if (iter != m_headParameters.end()) return iter->second;
                    return "Undefined!";
                }
            }
            else {
                return m_options.GetParameter(p_section, p_param);
            }
        }

        // Add insert entry to persistent buffer
        template <typename T>
        ErrorCode Index<T>::AddIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension,
                                     std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex,
                                     bool p_normalized)
        {
            if ((!m_options.m_useKV &&!m_options.m_useSPDK) || m_extraSearcher == nullptr) {
                LOG(Helper::LogLevel::LL_Error, "Only Support KV Extra Update\n");
                return ErrorCode::Fail;
            }

            if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0) return ErrorCode::EmptyData;
            if (p_dimension != GetFeatureDim()) return ErrorCode::DimensionSizeMismatch;

            SizeType begin, end;
            {
                std::lock_guard<std::mutex> lock(m_dataAddLock);

                begin = m_versionMap.GetVectorNum();
                end = begin + p_vectorNum;

                if (begin == 0) { return ErrorCode::EmptyIndex; }

                if (m_versionMap.AddBatch(p_vectorNum) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Info, "MemoryOverFlow: VID: %d, Map Size:%d\n", begin, m_versionMap.BufferSize());
                    exit(1);
                }

                if (m_pMetadata != nullptr) {
                    if (p_metadataSet != nullptr) {
                        m_pMetadata->AddBatch(*p_metadataSet);
                        if (HasMetaMapping()) {
                            for (SizeType i = begin; i < end; i++) {
                                ByteArray meta = m_pMetadata->GetMetadata(i);
                                std::string metastr((char*)meta.Data(), meta.Length());
                                UpdateMetaMapping(metastr, i);
                            }
                        }
                    }
                    else {
                        for (SizeType i = begin; i < end; i++) m_pMetadata->Add(ByteArray::c_empty);
                    }
                }
            }

            std::shared_ptr<VectorSet> vectorSet;
            if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized) {
                ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
                memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
                vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
                int base = COMMON::Utils::GetBase<T>();
                for (SizeType i = 0; i < p_vectorNum; i++) {
                    COMMON::Utils::Normalize((T*)(vectorSet->GetVector(i)), p_dimension, base);
                }
            }
            else {
                vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                    GetEnumValueType<T>(), p_dimension, p_vectorNum));
            }

            return m_extraSearcher->AddIndex(vectorSet, m_index, begin);
        }

        template <typename T>
        ErrorCode Index<T>::DeleteIndex(const SizeType &p_id)
        {
            if (m_versionMap.Delete(p_id)) return ErrorCode::Success;
            return ErrorCode::VectorNotFound;
        }

        template <typename T>
        ErrorCode Index<T>::DeleteIndex(const void* p_vectors, SizeType p_vectorNum)
        {
            // TODO: Support batch delete
            DimensionType p_dimension = GetFeatureDim();
            std::shared_ptr<VectorSet> vectorSet;
            if (m_options.m_distCalcMethod == DistCalcMethod::Cosine) {
                ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
                memcpy(arr.Data(), p_vectors, sizeof(T) * p_vectorNum * p_dimension);
                vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
                int base = COMMON::Utils::GetBase<T>();
                for (SizeType i = 0; i < p_vectorNum; i++) {
                    COMMON::Utils::Normalize((T*)(vectorSet->GetVector(i)), p_dimension, base);
                }
            }
            else {
                vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)p_vectors, sizeof(T) * 1 * p_dimension, false),
                    GetEnumValueType<T>(), p_dimension, 1));
            }
            SizeType p_id = m_extraSearcher->SearchVector(vectorSet, m_index);
            if (p_id == -1) return ErrorCode::ExternalAbort;

            return DeleteIndex(p_id);
        }
    }
}

#define DefineVectorValueType(Name, Type) \
template class SPTAG::SPANN::Index<Type>; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType


