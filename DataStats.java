import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.*;
import java.util.jar.Attributes;

//Weka libs
import Jama.Matrix;
import edu.princeton.cs.algs4.DirectedEdge;
import edu.princeton.cs.algs4.Edge;
import edu.princeton.cs.algs4.EdgeWeightedDigraph;
import edu.princeton.cs.algs4.StdOut;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;

public class DataStats {

    HashMap<String, Integer> wordsPolarity;
    ArrayList<SentenceStats> testSen;
    ArrayList<SentenceStats> trainSen;

    List<SentenceStats> testSenResult;


    int highest_PTrainID;
    int highest_PTrainCount;
    int highest_NTrainID;
    int highest_NTrainCount;

    int highest_PTestID;
    int highest_PTestCount;
    int highest_NTestID;
    int highest_NTestCount;


    static int totalTrainDoc = 50;
    static int totalTestDoc = 10;

    //For part II use
    Instances trainingInstances;
    Instances testingInstances;
    Classifier cModel;


    public DataStats()
    {
        wordsPolarity = new HashMap<String, Integer>();
        testSen = new ArrayList<SentenceStats>();
        trainSen = new ArrayList<SentenceStats>();
        testSenResult = new ArrayList<SentenceStats>();


        highest_PTrainID = -1;
        highest_PTrainCount = -1;
        highest_NTrainID = -1;
        highest_NTrainCount = -1;

        highest_PTestID = -1;
        highest_PTestCount = -1;
        highest_NTestID = -1;
        highest_NTestCount = -1;
    }

    void hashMapPinter()
    {
        Set set = wordsPolarity.entrySet();
        Iterator iterator = set.iterator();
        int i = 0;
        while  (iterator.hasNext()) {
            Map.Entry mentry = (Map.Entry) iterator.next();
            //System.out.print("words: " + mentry.getKey() + " Value: " + mentry.getValue() + "\n");
            i++;
        }
        System.out.print("num of Words: " + i + "\n");
        System.out.print("Hashmap size: " + wordsPolarity.size() + "\n");

    }

    void loadPolarityWords(String positivePolarityPath, String negativePolarityPath) throws IOException {

        //Do the positive words first
        BufferedReader br = new BufferedReader(new FileReader(positivePolarityPath));
        if(br == null)
        {
            System.out.printf("Unsuccessful read of the file\n");
            return;
        }
        String oneWord = null;

        while ((oneWord = br.readLine())!=null)
        {
            wordsPolarity.put(oneWord.toLowerCase(), 1);
        }

        //Do the negative words
        br = new BufferedReader(new FileReader(negativePolarityPath));
        while ((oneWord = br.readLine())!=null)
        {
            wordsPolarity.put(oneWord.toLowerCase(), -1);
        }
        hashMapPinter();
    }

    //Part 1 A of the project
    //test data has 0 to 9 files dir: test_data/test_files
    //train data has 0 to 49 files dir: train_data/train_files
    void computeDocumentStatistics(String trainingDataPath, String testDataPath) throws IOException {
        String oneSentence; //one sentence

        //Testing sentence statistics
        for(int docNum = 0; docNum < totalTestDoc; docNum++)
        {
            //Each document
            String testFilePath = testDataPath + "test_" + docNum + ".tsv";
            BufferedReader br = new BufferedReader(new FileReader(testFilePath));
            if(br.readLine() == null)
            {
                System.out.printf("Unsuccessful read of the file\n");
                return;
            }
            int tempNegativeCount = 0;
            int tempPositiveCount = 0;
            while ((oneSentence = br.readLine())!=null) //add for different files
            {
                //each sentence
                String [] senParts = oneSentence.split("\t");
                SentenceStats sentenceStats = sentenceProcessor(senParts, docNum, 0);
                testSen.add(sentenceStats);

                tempNegativeCount += sentenceStats.negativeWordCount;
                tempPositiveCount += sentenceStats.positiveWordCount;
            }

            //find the highest documents
            if (tempNegativeCount > highest_NTestCount)
            {
                highest_NTestCount = tempNegativeCount;
                highest_NTestID = docNum;
            }

            if (tempPositiveCount > highest_PTestCount)
            {
                highest_PTestCount = tempPositiveCount;
                highest_PTestID= docNum;
            }
        }

        //Training sentence statics
        for(int docNum = 0; docNum < totalTrainDoc; docNum++)
        {
            //each document
            String trainFilePath = trainingDataPath + "train_" + docNum + ".tsv";
            BufferedReader br = new BufferedReader(new FileReader(trainFilePath));
            if(br.readLine() == null)
            {
                System.out.printf("Unsuccessful read of the file\n");
                return;
            }

            int tempNegativeCount = 0;
            int tempPositiveCount = 0;
            while ((oneSentence = br.readLine())!=null) //add for different files
            {
                //each sentence
                String [] senParts = oneSentence.split("\t");


                //Add sentences

                SentenceStats sentenceStats = sentenceProcessor(senParts, docNum, 1);
                trainSen.add(sentenceStats);

                tempNegativeCount += sentenceStats.negativeWordCount;
                tempPositiveCount += sentenceStats.positiveWordCount;
            }

            //find the highest documents
            if (tempNegativeCount > highest_NTrainCount)
            {
                highest_NTrainCount = tempNegativeCount;
                highest_NTrainID = docNum;
            }

            if (tempPositiveCount > highest_PTrainCount)
            {
                highest_PTrainCount = tempPositiveCount;
                highest_PTrainID= docNum;
            }
        }
    }

    //dataOffset is for training, and if this is one, you check the sentence tag
    SentenceStats sentenceProcessor(String [] senParts, int docNum, int dataOffset)
    {
        SentenceStats sentenceStats = new SentenceStats();

        //document number
        sentenceStats.docNum = docNum;
        //sentence number
        sentenceStats.sentNum = Integer.parseInt(senParts[0]);

        //
        if(dataOffset == 1)
        {
            if(senParts[1].equals("O"))
            {
                //opinionated
                sentenceStats.isOpinionated = true;
            }
            else if(senParts[1].equals("F"))
            {
                //factual
                sentenceStats.isOpinionated = false;
            }
            else
            {
                System.out.print("No tag!!\n");
            }
        }

        //process the words in each sentence
        //parse each vocab
        String [] words = senParts[1+dataOffset].split("[\\s]+");
        for(int i = 0; i < words.length; i++)
        {
            if(wordsPolarity.containsKey(words[i].toLowerCase()))
            {
                  if(wordsPolarity.get(words[i].toLowerCase()) == 1)
                  {
                      sentenceStats.positiveWordCount = sentenceStats.positiveWordCount + 1;
                  }
                  else
                  {
                      sentenceStats.negativeWordCount = sentenceStats.negativeWordCount+ 1;
                  }
            }
        }

        //polarity of root
        if(wordsPolarity.containsKey(senParts[2+dataOffset].toLowerCase()))
        {
            if(wordsPolarity.get(senParts[2+dataOffset].toLowerCase()) == 1)
            {
                sentenceStats.polarityOfRoot = 1;
            }
            else
            {
                sentenceStats.polarityOfRoot = -1;
            }
        }
        else
        {
            sentenceStats.polarityOfRoot = 0;
        }

        //advMod
        sentenceStats.advMod = Integer.parseInt(senParts[3+dataOffset]);
        //aComp
        sentenceStats.aComp = Integer.parseInt(senParts[4+dataOffset]);
        //xComp
        sentenceStats.xComp = Integer.parseInt(senParts[5+dataOffset]);

        return sentenceStats;
    }

    void printTrainingAndTestingStat()
    {
        System.out.print("Most Positive Doc in Training:" + highest_PTrainID + "\tNumber of positive word: " + highest_PTrainCount + "\n");
        System.out.print("Most Negative Doc in Training:" + highest_NTrainID + "\tNumber of negative word: " + highest_NTrainCount + "\n");

        System.out.print("Most Positive Doc in Test:" + highest_PTestID + "\tNumber of positive word: " + highest_PTestCount + "\n");
        System.out.print("Most Negative Doc in Test:" + highest_NTestID + "\tNumber of negative word: " + highest_NTestCount + "\n");
    }

    void naiveBayesTraining() throws Exception {
        // Setting up Attributes
        Attribute classAtt = new Attribute("Opinionated Or Factual", new ArrayList<String>(Arrays.asList(new String[] {"O", "F"})));
        Attribute positivePolarWordCount = new Attribute("Positive Polar Word Count");
        Attribute negativePolarWordCount = new Attribute("Negative Polar Word Count");
        Attribute rootDependencyOfTree = new Attribute("Root Dependencey of Tree", new ArrayList<String>(Arrays.asList(new String[] {"-1", "0", "1"})));
        Attribute advMod = new Attribute("Presence of advMod", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
        Attribute aComp = new Attribute("Presence of aComp", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
        Attribute xComp = new Attribute("Presence of xComp", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));

        ArrayList<Attribute> wekaAttributes = new ArrayList<Attribute>();
        wekaAttributes.add(classAtt);
        wekaAttributes.add(positivePolarWordCount);
        wekaAttributes.add(negativePolarWordCount);
        wekaAttributes.add(rootDependencyOfTree);
        wekaAttributes.add(advMod);
        wekaAttributes.add(aComp);
        wekaAttributes.add(xComp);

        Instances isTrainingSet = new Instances("OpinionatedOrNot", wekaAttributes, trainSen.size());
        isTrainingSet.setClassIndex(0); //the first one is the class attribute

        fillUptheInstances(isTrainingSet, trainSen, wekaAttributes,7);
        trainingInstances = isTrainingSet;

        // Create a naïve bayes classifier
        cModel = (Classifier)new NaiveBayes();
        cModel.buildClassifier(isTrainingSet);
        evaluateClassificationModel(cModel, isTrainingSet);

    }

    //Filling up the training instances
    void fillUptheInstances(Instances isTrainingSet, ArrayList<SentenceStats> trainSen, ArrayList<Attribute> wekaAttributes,int numOfAttributes)
    {
        int capacity = trainSen.size(); //all sentences in the training set

        //every sentence
        for(int i = 0; i < capacity; i++)
        {
            //sentence object
            String biase;
            int numPositivePolar;
            int numNegativePolar;
            String rootDependency;
            String advMod;
            String aComp;
            String xComp;

            SentenceStats tempSen = trainSen.get(i);
            if(tempSen.isOpinionated == true)
            {
                biase = "O";
            }
            else
            {
                biase = "F";
            }
            numPositivePolar = tempSen.positiveWordCount;
            numNegativePolar = tempSen.negativeWordCount;

            if(tempSen.polarityOfRoot == -1)
            {
                rootDependency = "-1";
            }
            else if (tempSen.polarityOfRoot == 1)
            {
                rootDependency = "1";
            }
            else
            {
                rootDependency = "0";
            }

            if(tempSen.advMod == 1)
            {
                advMod = "1";
            }
            else
            {
                advMod = "0";
            }

            if(tempSen.aComp == 1)
            {
                aComp = "1";
            }
            else
            {
                aComp = "0";
            }

            if(tempSen.xComp == 1)
            {
                xComp = "1";
            }
            else
            {
                xComp = "0";
            }


            //each instance
            Instance oneSenInstance = new DenseInstance(numOfAttributes);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(0), biase);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(1), numPositivePolar);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(2), numNegativePolar);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(3), rootDependency);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(4), advMod);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(5), aComp);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(6), xComp);

            isTrainingSet.add(oneSenInstance);
        }
    }



    void evaluateClassificationModel(Classifier cModel, Instances data) throws Exception {
        System.out.print(cModel);
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(cModel, data);
        System.out.println(eval.toMatrixString());



    }


    //Part C
    void naiveBaseTesting(Classifier cModel) throws Exception {

        // Setting up Attributes
        Attribute classAtt = new Attribute("Opinionated Or Factual", new ArrayList<String>(Arrays.asList(new String[] {"O", "F"})));
        Attribute positivePolarWordCount = new Attribute("Positive Polar Word Count");
        Attribute negativePolarWordCount = new Attribute("Negative Polar Word Count");
        Attribute rootDependencyOfTree = new Attribute("Root Dependencey of Tree", new ArrayList<String>(Arrays.asList(new String[] {"-1", "0", "1"})));
        Attribute advMod = new Attribute("Presence of advMod", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
        Attribute aComp = new Attribute("Presence of aComp", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
        Attribute xComp = new Attribute("Presence of xComp", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));

        ArrayList<Attribute> wekaAttributes = new ArrayList<Attribute>();
        wekaAttributes.add(classAtt);
        wekaAttributes.add(positivePolarWordCount);
        wekaAttributes.add(negativePolarWordCount);
        wekaAttributes.add(rootDependencyOfTree);
        wekaAttributes.add(advMod);
        wekaAttributes.add(aComp);
        wekaAttributes.add(xComp);

        Instances isTestingSet = new Instances("unlabeled Test Set", wekaAttributes, testSen.size());
        isTestingSet.setClassIndex(0); //the first one is the class attribute
        fillUpTestInstances(isTestingSet, testSen, wekaAttributes, 7);

        testingInstances = isTestingSet;
        //test the instances
        for(int i = 0; i < isTestingSet.numInstances(); i++)
        {
            double [] instanceDistr = cModel.distributionForInstance(isTestingSet.instance(i));
            //update each sentence
            SentenceStats oneSentence = testSen.get(i);
            oneSentence.opinionProb = instanceDistr[0];
            oneSentence.factualProb = instanceDistr[1];

            //update the test sentence as well
            testSen.set(i, oneSentence);
            testSenResult.add(oneSentence);
        }

        //sort the result
        Collections.sort(testSenResult, SentenceStats.OpinionProbComparator);
        int topNum = 1;
        int docuNum = 10;
       SentenceStats [] topSentenceInEachDoc  = new SentenceStats[docuNum];
       boolean [] isDocFilled = new boolean[docuNum];
       for(int i = 0; i < docuNum; i++) isDocFilled[i] = false;
        //top 10 documents
        System.out.printf("==========================Top Opinionated Sentences from Each Document==========================\n");
        for(int i = 0; i < testSenResult.size(); i++)
        {

            if(isDocFilled[testSenResult.get(i).docNum])
            {
                //doNothing
            }
            else
            {
                topSentenceInEachDoc[testSenResult.get(i).docNum] = testSenResult.get(i);
                isDocFilled[testSenResult.get(i).docNum] = true;
            }
        }

       for(int i = 0; i<docuNum; i++)
       {
           topSentenceInEachDoc[i].printSentence();
       }

    }

    //the number of attributes should be 6, because of unlabeled data
    void fillUpTestInstances(Instances isTestingSet, ArrayList<SentenceStats> testSen, ArrayList<Attribute> wekaAttributes,int numOfAttributes )
    {
        int capacity = testSen.size(); //all sentences in the training set
        //every sentence
        for(int i = 0; i < capacity; i++)
        {
            //sentence object
            int numPositivePolar;
            int numNegativePolar;
            String rootDependency;
            String advMod;
            String aComp;
            String xComp;

            SentenceStats tempSen = testSen.get(i);
            numPositivePolar = tempSen.positiveWordCount;
            numNegativePolar = tempSen.negativeWordCount;


            if(tempSen.polarityOfRoot == -1)
            {
                rootDependency = "-1";
            }
            else if (tempSen.polarityOfRoot == 1)
            {
                rootDependency = "1";
            }
            else
            {
                rootDependency = "0";
            }

            if(tempSen.advMod == 1)
            {
                advMod = "1";
            }
            else
            {
                advMod = "0";
            }

            if(tempSen.aComp == 1)
            {
                aComp = "1";
            }
            else
            {
                aComp = "0";
            }

            if(tempSen.xComp == 1)
            {
                xComp = "1";
            }
            else
            {
                xComp = "0";
            }

            //each instance
            Instance oneSenInstance = new DenseInstance(numOfAttributes);   //you leave out the class label
            oneSenInstance.setValue((Attribute)wekaAttributes.get(1), numPositivePolar);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(2), numNegativePolar);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(3), rootDependency);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(4), advMod);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(5), aComp);
            oneSenInstance.setValue((Attribute)wekaAttributes.get(6), xComp);

            isTestingSet.add(oneSenInstance);
        }
    }

    //==========================================================================================================================================================================================
    //Part II
    //Part B
    void MatricesForAllDoc(ArrayList<EdgeWeightedDigraph> docGraphs_all, ArrayList<ArrayList<SentenceStats>> docsCollectStats)
    {
        ArrayList<Matrix> matricesForAllDoc = new ArrayList<>();
        for(int i = 0; i < docGraphs_all.size(); i++)
        {
           //add each document matrix to the array list
            matricesForAllDoc.add(constructWeightMatrix(docGraphs_all.get(i)));
        }

        //Calculate the matrix
        //setup the authority and hub matrix
        ArrayList<Matrix> authVecs = new ArrayList<>();
        ArrayList<Matrix> hubVecs = new ArrayList<>();
        for(int i = 0; i < docsCollectStats.size(); i++)
        {
            authVecs.add(setUpAuthMatrix(docsCollectStats.get(i)));
            hubVecs.add(setUpHubMatrix(docsCollectStats.get(i)));
        }

        //Perform multiplication
        int iterationNum = 10000;
        double epsilon = 0.00000000001;

        for(int i = 0; i < matricesForAllDoc.size(); i++)
        {
            //each document
            Matrix weightedMatrix = matricesForAllDoc.get(i);
            Matrix onehubVec = hubVecs.get(i);
            Matrix oneauthVec = authVecs.get(i);

            //normalize
            double [][] initialMatrix = {{1}, {1}};
            Matrix onehubVecNorm = new Matrix(initialMatrix);
            Matrix oneauthVecNorm = new Matrix(initialMatrix);

            //every iteration
            for(int j = 0; j < iterationNum; j++)
            {
                 onehubVecNorm = (weightedMatrix.transpose()).times(oneauthVec);
                 oneauthVecNorm = weightedMatrix.times(onehubVec);
                //Normalize the vector
                 onehubVecNorm = onehubVecNorm.times(1.0 / onehubVecNorm.normF());
                 oneauthVecNorm = oneauthVecNorm.times(1.0 / oneauthVecNorm.normF());

                //check the convergence factor
                Matrix authDifference = oneauthVec.minus(oneauthVecNorm);
                Matrix hubDifference = onehubVec.minus(onehubVecNorm);
                //if(differenceChecker(authDifference, epsilon) && differenceChecker(hubDifference, epsilon))
                //{
                //    System.out.printf("Test Doc " + i + "\n");
                //    break;
                //}

                //update the matrix
                 ArrayList<SentenceStats> tempDocStat = docsCollectStats.get(i);
                 updateWeigtedMatrix(weightedMatrix, oneauthVecNorm, onehubVecNorm, tempDocStat);
                 docsCollectStats.set(i, tempDocStat);

                 //update the vector
                 oneauthVec = oneauthVecNorm;
                 onehubVec = onehubVecNorm;
            }

            //Select the max
            int mostOpSen = mostOpinionatedSentence(onehubVecNorm);
            calc_Mscore(docsCollectStats.get(i), oneauthVecNorm, mostOpSen);


        }

    }

    void calc_Mscore(ArrayList<SentenceStats> oneDocStats, Matrix oneauthVecNorm, int i)
    {

        double maxMscore = 0;
        double secondMaxScore = 0;
        int maxEdgeValue = -1;
        int secondMaxEdgeValue = -1;
        //find the second max
            for(int j = 0; j < oneDocStats.size(); j++)
            {
                if(i != j)
                {
                    double sim_i_j = simCalc(i, j, oneDocStats);
                    //m score
                    double mSocre = sim_i_j * oneauthVecNorm.get(j, 0);
                    if(mSocre > maxMscore)
                    {
                        maxMscore = mSocre;
                        maxEdgeValue = j;
                    }
                }
            }

        for(int j = 0; j < oneDocStats.size(); j++)
        {
            if(i != j)
            {
                double sim_i_j = simCalc(i, j, oneDocStats);
                //m score
                double mSocre = sim_i_j * oneauthVecNorm.get(j, 0);
                if((mSocre != maxMscore) && (mSocre > secondMaxScore))
                {
                    secondMaxScore = mSocre;
                    secondMaxEdgeValue = j;
                }
            }
        }
        int j1base1 = maxEdgeValue + 1;
        int j2base1 = secondMaxEdgeValue + 1;
        System.out.printf("Top Supporting Sentence: " + j1base1 + " M score: " + maxMscore + "\n");
        System.out.printf("Top Supporting Sentence: " + j2base1 + " M score: " + secondMaxScore+ "\n\n");
    }

    int mostOpinionatedSentence(Matrix onehubVecNorm)
    {
        int mostOpSen = -1;
        double highestHubscore = -1;
        for(int i = 0; i < onehubVecNorm.getRowDimension(); i++)
        {
            if(onehubVecNorm.get(i,0) > highestHubscore)
            {
                highestHubscore = onehubVecNorm.get(i,0);
                mostOpSen = i;
            }
        }

        int base1DocNum = mostOpSen + 1;
        System.out.printf("Most Opinionated sentence: " + base1DocNum + "\thub score: " + highestHubscore+"\n");
        return mostOpSen;
    }



    boolean differenceChecker(Matrix diffMatrix, double epsilon)
    {
        for(int i = 0; i < diffMatrix.getRowDimension(); i++)
        {
            if(Math.abs(diffMatrix.get(i,0)) > epsilon)
            {
                return false;
            }
        }
        return true;
    }


    Matrix constructWeightMatrix(EdgeWeightedDigraph oneDocGraph)
    {
        double [][] weightedMatrix = new double[oneDocGraph.V()][oneDocGraph.V()];
        //construct an edgeWeighted matrix
        for(int j = 0; j < oneDocGraph.V(); j++)
        {
            //each vertex
            Iterable<DirectedEdge> edgeAdjToi = oneDocGraph.adj(j);
            for(DirectedEdge e: edgeAdjToi)
            {
                if(j != e.from()) System.out.printf("index does Not match up\n\n\n"); //this should not be printed

                weightedMatrix[e.to()][j] = e.weight();
            }
            weightedMatrix[j][j] = 0;
        }

        Matrix w = Matrix.constructWithCopy(weightedMatrix);
        return w;
    }

    void updateWeigtedMatrix(Matrix weightedMatrix, Matrix oneauthVecNorm, Matrix onehubVecNorm, ArrayList<SentenceStats> oneDocStats)
    {
        //update hub scores and authority scores for sentence stats
        for(int i = 0; i < oneDocStats.size(); i++)
        {
            SentenceStats oneSen = oneDocStats.get(i);
            oneSen.opinionProb = onehubVecNorm.get(i, 0);
            oneSen.factualProb = oneauthVecNorm.get(i, 0);
            oneDocStats.set(i, oneSen);
        }

        //update the matrix
//        for(int i = 0; i < oneDocStats.size(); i++)
//        {
//            for(int j = 0; j < oneDocStats.size(); j++)
//            {
//                if(i != j)
//                {
//                    double sim_i_j = simCalc(i, j, oneDocStats);
//                    //edge weight equation
//                    double weight_i_j = Math.pow(sim_i_j, 2) * Math.pow(oneDocStats.get(i).opinionProb, 3) * (1.0 + 1.0 / Math.abs(i - j));
//                    weightedMatrix.set(i, j, weight_i_j);
//                }
//            }
//        }

    }

    Matrix setUpAuthMatrix(ArrayList<SentenceStats> oneDocStat)
    {
        double [][] authVectorOneDoc = new double[oneDocStat.size()][1];
        for(int i = 0; i < oneDocStat.size(); i++)
        {
            //each sentence
            authVectorOneDoc[i][0] = oneDocStat.get(i).factualProb;
        }
        return Matrix.constructWithCopy(authVectorOneDoc);
    }

    Matrix setUpHubMatrix(ArrayList<SentenceStats> oneDocStat)
    {
        double [][] hubVectorOneDoc = new double[oneDocStat.size()][1];
        for(int i = 0; i < oneDocStat.size(); i++)
        {
            //add up the hub score
            hubVectorOneDoc[i][0] = oneDocStat.get(i).opinionProb;
        }
        return Matrix.constructWithCopy(hubVectorOneDoc);
    }


    //Part A the location of the cModel
    //This is the overall function for part A
    void buildupNodesForEveryDoc(Classifier cModel, Instances trainingInstances, ArrayList<SentenceStats> trainSen, String folderPath, int numOfDoc, String filePrefix) throws Exception {

        //Evaluate everyone
        for(int i = 0; i < trainingInstances.numInstances(); i++)
        {
            double [] instanceDistr = cModel.distributionForInstance(trainingInstances.instance(i));            //update each sentence
            trainSen.get(i).opinionProb = instanceDistr[0];
            trainSen.get(i).factualProb = instanceDistr[1];
        }
        ArrayList<ArrayList<SentenceStats>> trainSenByDoc = divideTrainSenByDoc(trainSen);
        //load up the document
        trainSenByDoc = readDocVectors(folderPath, numOfDoc, trainSenByDoc, filePrefix); //this part should be changed for testing


        ArrayList<EdgeWeightedDigraph> docGraphs_train = new ArrayList<>();            //it would be different for testing
        for(int i = 0; i < trainSenByDoc.size(); i++)
        {
            //each document: constructing a directed graph
            ArrayList<SentenceStats> oneDocStats = trainSenByDoc.get(i);
            EdgeWeightedDigraph docNodes = new EdgeWeightedDigraph(oneDocStats.size());
            fillUpNode(docNodes, oneDocStats);
            docGraphs_train.add(docNodes);
        }

        MatricesForAllDoc( docGraphs_train, trainSenByDoc);

    }

    //Constructing a directed graph for each document
    void fillUpNode(EdgeWeightedDigraph docNodes, ArrayList<SentenceStats> oneDocStats)
    {
        for(int i = 0; i < oneDocStats.size(); i++)
        {
            for(int j = 0; j < oneDocStats.size(); j++)
            {
                if(i != j)
                {
                    double sim_i_j = simCalc(i, j, oneDocStats);
                    //edge weight equation
                    double weight_i_j = Math.pow(sim_i_j, 2) * Math.pow(oneDocStats.get(i).opinionProb, 3) * (1.0 + 1.0 / Math.abs(i - j));
                    DirectedEdge tempEdge = new DirectedEdge(i, j,weight_i_j);
                    docNodes.addEdge(tempEdge);
                }
            }
        }
    }

    //given two vertex, you calculate the their similarities
    double simCalc(int vertex1, int vertex2, ArrayList<SentenceStats> oneDocStats)
    {
        //trainSen.get(docNum).get(vertex1).senVector
        double dotProd = 0.0;
        double scale1 = 0.0;
        double scale2 = 0.0;

        for(int i = 0; i < (oneDocStats.get(vertex1).senVector).length; i++)
        {

            dotProd += (oneDocStats.get(vertex1)).senVector[i] * (oneDocStats.get(vertex2)).senVector[i];
            scale1 += Math.pow((oneDocStats.get(vertex1)).senVector[i], 2);
            scale2 += Math.pow((oneDocStats.get(vertex2)).senVector[i], 2);
        }
        if(dotProd == 0)
        {
            return 0;
        }
        double simVal = dotProd / (Math.sqrt(scale1) * Math.sqrt(scale2));
        return simVal;
    }

    ArrayList<ArrayList<SentenceStats>> readDocVectors(String folderPath, int numOfDoc, ArrayList<ArrayList<SentenceStats>> trainSenByDoc, String filePrefix) throws IOException {
        for(int i = 0; i < numOfDoc; i++)
        {
            //Each document
            String filePath = folderPath + filePrefix + i + ".tsv";
            BufferedReader br = new BufferedReader(new FileReader(filePath));

            if(br.readLine() == null) { System.out.printf("Unsuccessful read of the file\n");}

            ArrayList<SentenceStats> tempDoc = trainSenByDoc.get(i);
            String oneLine;
            int docCounter = 0;
            while((oneLine = br.readLine()) != null)
            {
               SentenceStats tempSen = tempDoc.get(docCounter);
               tempSen.senVector = breakStringIntoVectors(oneLine);
               tempDoc.set(docCounter, tempSen);
               docCounter++;    //update the counter
            }
            trainSenByDoc.set(i, tempDoc);
        }

        return trainSenByDoc;
    }

    //parsing vector documents
    double[] breakStringIntoVectors(String oneLine)
    {
        String [] vectorLine = oneLine.split("\t");
        String [] numberVector = vectorLine[1].split(", ");
        double [] senVector = new double[numberVector.length];

        double tempDouble;
        for(int i = 0; i <numberVector.length; i++)
        {
            if(i == 0)
            {
                //first vector
                tempDouble = Double.parseDouble(numberVector[i].substring(1));
            }
            else if(i == (numberVector.length -1))
            {
                //last vector
                int lastIndex = numberVector[i].length() - 2;
                tempDouble = Double.parseDouble(numberVector[i].substring(0, lastIndex));
            }
            else
            {
                tempDouble = Double.parseDouble(numberVector[i]);
            }
            senVector[i] = tempDouble;
        }

        return senVector;
    }


    //group sentence stat into different documents
    ArrayList<ArrayList<SentenceStats>> divideTrainSenByDoc(ArrayList<SentenceStats> totalSen)
    {
        //Process trainSen into different documents
        ArrayList<ArrayList<SentenceStats>> trainSenByDoc = new ArrayList<>();
        int docNum = -1; // this will change
        ArrayList<SentenceStats> oneDocData = new ArrayList<>();
        for(int i = 0; i <totalSen.size();i++)
        {
            if(docNum != totalSen.get(i).docNum)
            {
                //next document
                if(docNum != -1)
                {
                    trainSenByDoc.add(oneDocData);
                }
                oneDocData = new ArrayList<>();
                docNum = totalSen.get(i).docNum;
                oneDocData.add(totalSen.get(i));
            }
            else
            {
                oneDocData.add(totalSen.get(i));
            }
        }
        trainSenByDoc.add(oneDocData);
        return trainSenByDoc;
    }

    ArrayList<Attribute> setUpWekaArribute()
    {
        // Setting up Attributes
        Attribute classAtt = new Attribute("Opinionated Or Factual", new ArrayList<String>(Arrays.asList(new String[] {"O", "F"})));
        Attribute positivePolarWordCount = new Attribute("Positive Polar Word Count");
        Attribute negativePolarWordCount = new Attribute("Negative Polar Word Count");
        Attribute rootDependencyOfTree = new Attribute("Root Dependencey of Tree", new ArrayList<String>(Arrays.asList(new String[] {"-1", "0", "1"})));
        Attribute advMod = new Attribute("Presence of advMod", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
        Attribute aComp = new Attribute("Presence of aComp", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
        Attribute xComp = new Attribute("Presence of xComp", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));

        ArrayList<Attribute> wekaAttributes = new ArrayList<Attribute>();
        wekaAttributes.add(classAtt);
        wekaAttributes.add(positivePolarWordCount);
        wekaAttributes.add(negativePolarWordCount);
        wekaAttributes.add(rootDependencyOfTree);
        wekaAttributes.add(advMod);
        wekaAttributes.add(aComp);
        wekaAttributes.add(xComp);

        return wekaAttributes;
    }



//Part II above
    public static void main(String[] args) throws Exception {
        DataStats dataStats = new DataStats();
        dataStats.loadPolarityWords("pos_polar.txt", "neg_polar.txt");
        dataStats.computeDocumentStatistics("train_data/train_files/", "test_data/test_files/");
        dataStats.printTrainingAndTestingStat();

        //Training
        dataStats.naiveBayesTraining();

        //classifying
        dataStats.naiveBaseTesting(dataStats.cModel);

        System.out.printf("\n\n\n=======================Part 2======================\n");
        //System.out.printf("=======================training doc======================\n");
        //dataStats.buildupNodesForEveryDoc(dataStats.cModel, dataStats.trainingInstances, dataStats.trainSen, "train_data/train_vectors/", 50, "train_vec_");

        System.out.printf("\n\n\n=======================testing doc======================\n");
        dataStats.buildupNodesForEveryDoc(dataStats.cModel, dataStats.testingInstances, dataStats.testSen, "test_data/test_vectors/", 10, "test_vec_");

    }

}
