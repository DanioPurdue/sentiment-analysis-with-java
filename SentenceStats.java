import java.util.Comparator;

public class SentenceStats {
    int sentNum;             //record the sent num
    int docNum;              //record the document num
    int positiveWordCount;
    int negativeWordCount;
    int polarityOfRoot;
    int advMod;
    int aComp;
    int xComp;
    boolean isOpinionated;

    double opinionProb;
    double factualProb;

    double [] senVector;

    void printSentence()
    {
        System.out.print("=========================================================\n");
        System.out.print("Sentence Number: " + sentNum);
        System.out.print(" Document Number: " + docNum + "\n");
        //System.out.print("Positive Word Count: " + positiveWordCount + "\n");
        //System.out.print("Negative Word Count: " + negativeWordCount + "\n");
        //System.out.print("Polarity of Root: " + polarityOfRoot + "\n");
        //System.out.print("advMod: " + advMod + "\n");
        //System.out.print("aComp: " + aComp + "\n");
        //System.out.print("xComp: " + xComp + "\n");
        System.out.print("Opinionated Probability: " + opinionProb + "\t");
        System.out.print("Factual Probability: " + factualProb + "\n");
    }

    public SentenceStats()
    {
        sentNum = 0;
        docNum = 0;
        positiveWordCount = 0;
        negativeWordCount = 0;
        polarityOfRoot = 0;
        advMod = 0;
        aComp = 0;
        xComp = 0;
        isOpinionated = false;

        opinionProb = 0.0;
        factualProb = 0.0;
        senVector = new double[300]; //300 numbers
    }

    public static Comparator<SentenceStats> OpinionProbComparator = new Comparator<SentenceStats>() {
        @Override
        public int compare(SentenceStats o1, SentenceStats o2) {
            return -Double.compare(o1.opinionProb, o2.opinionProb);
        }
    };

    public static Comparator<SentenceStats> FactualProbComparator = new Comparator<SentenceStats>() {
        @Override
        public int compare(SentenceStats o1, SentenceStats o2) {
            return -Double.compare(o1.factualProb, o2.factualProb);
        }
    };
}

