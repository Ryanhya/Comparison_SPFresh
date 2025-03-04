// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_POSTINGVERSIONLABEL_H_
#define _SPTAG_COMMON_POSTINGVERSIONLABEL_H_

#include <atomic>
#include "Dataset.h"

namespace SPTAG
{
    namespace COMMON
    {
        enum class status : std::uint64_t{
            normal = 0x00,
            splitting = 0x01,
            merging = 0x02,
            deleted = 0x03
        };

        class PostingVersionLabel
        {
        private:
            std::atomic<SizeType> m_deleted;//32 bits atomic signed integer
            Dataset<std::uint64_t> m_data;//uint64_t is 8 bytes
            std::mutex m_countLock;

            const std::uint64_t status_range = 0x0000000000000003;//[1:0] is valid, 00 is normal,  01 is splitting, 10 is merging, 11 is deleted
            const std::uint64_t count_range = 0x00000000000003fc;//[9:2] is valid
            const std::uint64_t PID2_range = 0x0000001ffffffc00;//[36:10] is valid
            const std::uint64_t PID1_range = 0xffffffe000000000;//[63:37] is valid
            
        public:
            PostingVersionLabel() 
            {
                m_deleted = 0;
                m_data.SetName("postingVersionLabelID");
            }

            void Initialize(SizeType size, SizeType blockSize, SizeType capacity)
            {
                m_data.Initialize(size, 1, blockSize, capacity);
            }

            inline size_t Count() const { return m_data.R() - m_deleted.load(); }

            inline size_t GetDeleteCount() const { return m_deleted.load();}

            inline bool Merged(const SizeType& key) const
            {
                return (*m_data[key] & status_range) == static_cast<uint64_t>(status::merging);
            }

            inline bool Merge(const SizeType& key)
            {
                uint64_t v = *m_data[key] & static_cast<uint64_t>(status::merging);
                uint64_t oldvalue = (uint64_t)InterlockedExchange8((char*)(m_data[key]), (char)v);
                if (oldvalue == v) return false;
                return true;
            }

            inline bool Splitted(const SizeType& key) const
            {
                return (*m_data[key] & status_range) == static_cast<uint64_t>(status::splitting);
            }

            inline bool Split(const SizeType& key)
            {
                uint64_t v = *m_data[key] & static_cast<uint64_t>(status::splitting);
                uint64_t oldvalue = (uint64_t)InterlockedExchange8((char*)(m_data[key]), (char)v);
                if (oldvalue == v) return false;
                return true;
            }

            inline bool Deleted(const SizeType& key) const
            {
                return (*m_data[key] & status_range) == static_cast<uint64_t>(status::deleted);
            }

            inline bool Delete(const SizeType& key)
            {
                uint64_t v = *m_data[key] & static_cast<uint64_t>(status::deleted);
                uint64_t oldvalue = (uint64_t)InterlockedExchange8((char*)(m_data[key]), (char)v);
                if (oldvalue == v) return false;
                m_deleted++;
                return true;
            }

            inline uint64_t GetData(const SizeType& key)
            {
                return *m_data[key];
            }

            inline uint64_t GetCount(const SizeType& key)
            {
                std::lock_guard<std::mutex> lock(m_countLock);
                return (*m_data[key] & count_range);
            }

            inline bool IncCount(const SizeType& key, uint64_t* newCount)
            {
                std::lock_guard<std::mutex> lock(m_countLock);
                while (true) {
                    if (Deleted(key)) return false;
                    uint64_t oldCount = GetCount(key);
                    uint64_t move_oldCount = oldCount >> 2;

                    *newCount = (move_oldCount+1) << 2;
                    if (((uint64_t)InterlockedCompareExchange((char*)m_data[key], (char)*newCount, (char)move_oldCount)) == move_oldCount) {
                        return true;
                    }
                }
            }

            inline SizeType GetVectorNum()
            {
                return m_data.R();
            }

            inline ErrorCode Save(std::shared_ptr<Helper::DiskIO> output)
            {
                SizeType deleted = m_deleted.load();
                IOBINARY(output, WriteBinary, sizeof(SizeType), (char*)&deleted);
                return m_data.Save(output);
            }

            inline ErrorCode Save(const std::string& filename)
            {
                LOG(Helper::LogLevel::LL_Info, "Save %s To %s\n", m_data.Name().c_str(), filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Save(ptr);
            }

            inline ErrorCode Load(std::shared_ptr<Helper::DiskIO> input, SizeType blockSize, SizeType capacity)
            {
                SizeType deleted;
                IOBINARY(input, ReadBinary, sizeof(SizeType), (char*)&deleted);
                m_deleted = deleted;
                return m_data.Load(input, blockSize, capacity);
            }

            inline ErrorCode Load(const std::string& filename, SizeType blockSize, SizeType capacity)
            {
                LOG(Helper::LogLevel::LL_Info, "Load %s From %s\n", m_data.Name().c_str(), filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return Load(ptr, blockSize, capacity);
            }

            inline ErrorCode Load(char* pmemoryFile, SizeType blockSize, SizeType capacity)
            {
                m_deleted = *((SizeType*)pmemoryFile);
                return m_data.Load(pmemoryFile + sizeof(SizeType), blockSize, capacity);
            }

            inline ErrorCode AddBatch(SizeType num)
            {
                return m_data.AddBatch(num);
            }

            inline std::uint64_t BufferSize() const 
            {
                return m_data.BufferSize() + sizeof(SizeType);
            }

            inline void SetR(SizeType num)
            {
                m_data.SetR(num);
            }
        };
    }
}

#endif // _SPTAG_COMMON_LABELSET_H_
