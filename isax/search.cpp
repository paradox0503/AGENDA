#include "iSAX/isax_factory.hpp"
#include "random_data.h"
#include <iostream>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iomanip>
// #include <windows.h>
#include <utility>
#include <string>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>
#include <string_view> // 需包含此头文件
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <string>
#include <stdexcept>
std::string read_nth_line(const std::string& filename, int line_num) {
    // 参数验证
    if (line_num <= 0) {
        throw std::runtime_error("行号必须大于0");
    }

    // 打开文件
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    std::string line;
    int current_line = 1;

    // 逐行读取，直到目标行
    while (current_line < line_num && std::getline(file, line)) {
        current_line++;
    }

    // 读取目标行
    if (!std::getline(file, line)) {
        file.close();
        throw std::runtime_error("文件只有 " + std::to_string(current_line) + " 行");
    }

    file.close();
    return line;
}
using namespace std;

// 函数：将搜索结果写入文件
// 修改 writeResultsToFile 函数，添加一个控制是否覆盖的参数
void writeResultsToFile(size_t i,const std::string& filename, const std::vector<std::pair<float, uint64_t>>& results,
                       const std::string& dataset, bool overwrite = false) {
    string subfolder = "1stBSF";
    mkdir(subfolder.c_str(), 0777);

    // 根据 overwrite 标志决定打开模式
    std::ios_base::openmode mode = std::ios::out;
    if (overwrite) {
        mode |= std::ios::trunc;  // 覆盖模式
    } else {
        mode |= std::ios::app;    // 追加模式
    }

    std::ofstream outFile(subfolder + "/" + filename, mode);
    if (!outFile.is_open()) {
        std::cerr << "无法打开文件 " << filename << " 进行写入操作。" << std::endl;
        return;
    }
    outFile << std::fixed << std::setprecision(4);
    for (const auto& result : results) {
        outFile <<i<< "," << result.second << "," << result.first << std::endl;
    }
    outFile.close();
}


int main() {
    uint64_t query_num = std::stoi(read_nth_line("/data/AGENDA/isax/search.txt", 1));//+
    const uint64_t k = std::stoi(read_nth_line("/data/AGENDA/isax/search.txt", 2)); //+
    const string data_name = read_nth_line("/data/AGENDA/isax/search.txt", 3);  //+
    static const std::string origin_query_directory = read_nth_line("/data/AGENDA/isax/search.txt", 4);
    static const std::string origin_input_directory = origin_query_directory;//+
    static const std::string embed_query_directory = read_nth_line("/data/AGENDA/isax/search.txt", 5);
    static const std::string embed_input_directory = embed_query_directory;//+
    // 定义常量
    const string input_filename = origin_input_directory + data_name+"-dataset.bin"; // 数据集路径
    const string emb_input_filename = embed_input_directory +data_name+ "-database.bin"; // 数据集路径
    const string query_filename = origin_query_directory +data_name+ "-query.bin"; // 查询文件的路径
    const string emb_query_filename = embed_query_directory +data_name+ "-query.bin"; // 查询文件的路径
    const string output_directory = "./"+data_name+"_index/";
    const uint64_t sax_length = 16;
    uint32_t source_value =std::stoi(read_nth_line("/data/AGENDA/isax/search.txt", 6));//+
    const uint32_t ts_length = source_value;//
    const uint64_t ts_num = std::stoi(read_nth_line("/data/AGENDA/isax/search.txt", 7));//+


    // 加载查询数据
    FILE* query_file = fopen(query_filename.c_str(), "rb");
    if (!query_file) {
        cerr << "Failed to open query file: " << query_filename << endl;
        return -1;
    }
    FILE* emb_query_file = fopen(emb_query_filename.c_str(), "rb");
    if (!emb_query_file) {
        cerr << "Failed to open query file: " << emb_query_filename << endl;
        return -1;
    }

    // 读取所有查询数据
    vector<vector<float>> queries(query_num, vector<float>(ts_length));
    for (uint64_t i = 0; i < query_num; ++i) {
        size_t read_count = fread(static_cast<void*>(queries[i].data()), sizeof(float), ts_length, query_file);
        if (read_count != ts_length) {
            cerr << "Failed to read query data at index " << i << endl;
            fclose(query_file);
            return -1;
        }
    }
    fclose(query_file);

    vector<vector<float>> emb_queries(query_num, vector<float>(sax_length));
    for (uint64_t i = 0; i < query_num; ++i) {
        size_t read_count = fread(static_cast<void*>(emb_queries[i].data()), sizeof(float), sax_length, emb_query_file);
        if (read_count != sax_length) {
            cerr << "Failed to read query data at index " << i << endl;
            fclose(emb_query_file);
            return -1;
        }
    }
    fclose(emb_query_file);

    if(ts_length==256){
        auto isax_index = isax::iSAXFactory<256, sax_length>::createFromIndex(data_name, output_directory);
        vector<int32_t> search_nums = {-1};

        // 对每种搜索节点数量执行搜索
        for (int32_t search_max_num : search_nums) {
            std::string filename_e = data_name + ".txt";
            bool isFirstWrite = true;
            for (size_t i = 0; i < queries.size(); ++i) {
                string subfolder ="1stBSF";
                mkdir(subfolder.c_str(), 0777) ;
                const auto& query = queries[i];
                const auto& emb_query = emb_queries[i];
                auto appro_ans = isax_index->approximateSearch((void*)query.data(),(void*)emb_query.data(), k, 10,search_max_num);
                // 首次写入时覆盖，之后追加
                writeResultsToFile(i,filename_e, appro_ans, data_name, isFirstWrite);
                // 写入一次后将标志位设为 false
                isFirstWrite = false;
            }
        }

        // 释放资源
        delete isax_index;
    }else if(ts_length==128){
        auto isax_index = isax::iSAXFactory<128, sax_length>::createFromIndex(data_name, output_directory);
        vector<int32_t> search_nums = {-1};

        // 对每种搜索节点数量执行搜索
        for (int32_t search_max_num : search_nums) {
            std::string filename_e = data_name + ".txt";
            bool isFirstWrite = true;
            for (size_t i = 0; i < queries.size(); ++i) {
                string subfolder ="1stBSF";
                mkdir(subfolder.c_str(), 0777) ;
                const auto& query = queries[i];
                const auto& emb_query = emb_queries[i];
                auto appro_ans = isax_index->approximateSearch((void*)query.data(),(void*)emb_query.data(), k, 10,search_max_num);
                // 首次写入时覆盖，之后追加
                writeResultsToFile(i,filename_e, appro_ans, data_name, isFirstWrite);
                // 写入一次后将标志位设为 false
                isFirstWrite = false;
            }
        }

        // 释放资源
        delete isax_index;
    }else if(ts_length==96){
        auto isax_index = isax::iSAXFactory<96, sax_length>::createFromIndex(data_name, output_directory);
    vector<int32_t> search_nums = {-1};

        // 对每种搜索节点数量执行搜索
        for (int32_t search_max_num : search_nums) {
            std::string filename_e = data_name + ".txt";
            bool isFirstWrite = true;
            for (size_t i = 0; i < queries.size(); ++i) {
                string subfolder ="1stBSF";
                mkdir(subfolder.c_str(), 0777) ;
                const auto& query = queries[i];
                const auto& emb_query = emb_queries[i];
                auto appro_ans = isax_index->approximateSearch((void*)query.data(),(void*)emb_query.data(), k, 10,search_max_num);
                // 首次写入时覆盖，之后追加
                writeResultsToFile(i,filename_e, appro_ans, data_name, isFirstWrite);
                // 写入一次后将标志位设为 false
                isFirstWrite = false;
            }
        }

        // 释放资源
        delete isax_index;
    }
    return 0;
}