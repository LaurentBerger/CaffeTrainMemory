#include <caffe/caffe.hpp>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <fstream>

#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

using namespace std;
using namespace cv;

void main(void)
{
    vector<String> fileNames;
    // Structure N filtres par itération , M itération par fichiers, P réseaux filters[p][m][n]
    vector<vector<vector<Mat>>> filters;

    glob("conv1_06*.yml", fileNames);
    int nbIterRef = -1;
    int nbFilters = -1;
    for (auto file : fileNames)
    {
        cout << "File : " << file << endl;
        FileStorage fs;
        fs.open(file, FileStorage::READ);

        int idxStream = 0;
        FileNode n;
        vector<vector<Mat>> w;
        do
        {
            n = fs.root(idxStream++);
            if (!n.empty())
            {
                //                cout << n.name() <<" "<< n.type()<< endl;
                FileNodeIterator it = n.begin();
                vector<Mat> x;
                for (; it != n.end(); it++)
                {
                    //                    cout << (*it).name() << "\n";
                    fs[(*it).name()] >> x;
                }
                w.push_back(x);
            }
        } while (!n.empty());
        filters.push_back(w);

    }
    vector<vector<vector<int>>> corrFilter;
    /*! corrFilter[file1][indFiltre][file2] = le filtre indFiltre du fichier 1  est associé
    dans le fichier 2 au fichier corrFilter[file1][indFiltre][file2]*/
    for (int ra = 0; ra < filters.size(); ra++)
    {
        corrFilter.push_back(vector<vector<int>>());
        for (int idxFiltera = 0; idxFiltera < filters[ra][0].size(); idxFiltera++)
        {
            corrFilter[ra].push_back(vector<int>());
            for (int rb = 0; rb < filters.size(); rb++)
                corrFilter[ra][idxFiltera].push_back(-1);
        }
    }

    int nbStabilite = filters[0].size();
    for (int ra = 0; ra < filters.size(); ra++) // Parcours des fichiers
    {
        for (int itea = filters[ra].size() - 1; itea < filters[ra].size(); itea++) // Dernière itération
        {
            for (int idxFiltera = 0; idxFiltera < filters[ra][itea].size(); idxFiltera++) // filtre de reférence
            {
                //                cout << "File " << fileNames[ra] << " ** Ite " << itea << "** Filter " << idxFiltera << "\n";
                double x = mean(filters[ra][itea][idxFiltera])[0];
                double x2 = norm(filters[ra][itea][idxFiltera]);
                for (int rb = 0; rb < filters.size(); rb++) // parcours du fichier rb
                {
                    int idxFile = -1;
                    for (int iteb = filters[rb].size() - 1; iteb < filters[rb].size(); iteb++) // parcours des itération du fichier rb 
                    {
                        double m = -1;
                        int idxIte = -1;
                        for (int idxFilterb = 0; idxFilterb < filters[rb][iteb].size(); idxFilterb++)
                        {
                            double y = mean(filters[rb][iteb][idxFilterb])[0];
                            double y2 = norm(filters[rb][iteb][idxFilterb]);
                            Mat p;
                            multiply(filters[ra][itea][idxFiltera] - x, filters[rb][iteb][idxFilterb] - y, p);
                            y = sum(p)[0] / x2 / y2;
                            if (y > m)
                            {
                                m = y;
                                idxIte = idxFilterb;
                            }

                        }
                        //                        cout << idxIte << "\t";
                        if (idxFile == -1)
                            idxFile = idxIte;
                        else
                            if (idxFile != idxIte)
                                idxFile = -2;
                    }
                    corrFilter[ra][idxFiltera][rb] = idxFile;
                    //                    cout << "----> " << idxFile << "(" << fileNames[rb] << ")";
                    /*                    if (idxFile == -2)
                                            cout << "++++++++++++++++++++++\n";
                                        else
                                            cout << "\n";*/
                }
                //                cout << "\n**************************************************************\n";
            }
        }
        //        cout << "\n";
    }
    // Verification bijection 
    vector<vector<int>> bijection;
    for (int ra = 0; ra < corrFilter.size(); ra++) // Fichier
    {
        bijection.push_back(vector<int>(corrFilter.size()));
        for (int rb = 0; rb < corrFilter.size(); rb++) // verification de la bijetcion
        {
            bool bij = true;
            int nbBijection = 0;
            for (int idxFiltera = 0; idxFiltera < corrFilter[ra].size(); idxFiltera++)// filtre idxFiltera du fichier ra
            {
                int ind1 = corrFilter[ra][idxFiltera][rb]; // filtre idxFiltera du fichier ra ressemble au filtre ind1 du fichier rb
                if (ind1 >= 0)
                {
                    int ind2 = corrFilter[rb][ind1][ra]; // filtre ind1 du fichier rb ressemble au filtre ind2 du fichier ra 
                    if (ind2 == idxFiltera)
                    {
                        cout << "Fichier " << fileNames[ra] << "-Filtre " << idxFiltera << "-> Fichier " << fileNames[rb] << " avec " << ind1;
                        cout << "\n";
                        nbBijection++;
                    }
                    else
                    {
                        bij = false;
                        cout << "Fichier " << fileNames[ra] << "-Filtre " << idxFiltera << "-> Fichier " << fileNames[rb] << " : " << ind1 << " # " << corrFilter[rb][ind1][ra];
                        cout << "\n";
                    }
                }
                else
                {
                    bij = false;
                    cout << "Fichier " << fileNames[ra] << "-Filtre " << idxFiltera << "-> Fichier " << fileNames[rb];
                    cout << "*** NON STABLE**\n";
                }
            }
            bijection[ra][rb] = nbBijection;

        }
    }
    for (int ra = 0; ra < corrFilter.size(); ra++)
    {
        for (int rb = 0; rb < corrFilter.size(); rb++)
            cout << bijection[ra][rb] << "\t";
        cout << "\n";
    }
}
