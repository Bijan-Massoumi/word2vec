from ppmi import *


class QuoraWord2Vec:
    def __init__(self,data_set):
        lines = collect_sentences(data_set)
        (keys,ppmi_matrix) = compute_ppmi(lines)

        self.word_to_num = keys
        self.data_set = data_set
        self.lines = lines
        self.number_of_eig = 100
        self.embedded_matrix = self.create_embedding_matrix(ppmi_matrix,\
                                                    self.number_of_eig)
        (scores,answers) = self.compute_all_scores()
        print np.mean(scores)
        self.scores = scores
        self.answers = answers

    def create_embedding_matrix(self,ppmi,top_k):
        (vals,vectors) = eigen.eigs(ppmi,top_k)
        row = np.ones(top_k, dtype='uint32')
        col = np.ones(top_k, dtype='uint32')
        data = np.ones(top_k)
        print "constructing F and Sigma"
        #construct F
        F = linear.csr_matrix(vectors)
        #construct sigma
        for i in range(0,top_k):
            row[i] = i
            col[i] = i
            data[i] = vals[i]

        sigma = linear.csr_matrix( (data,(row,col)),\
                shape = (top_k,top_k))
        print "multiplying F and sigma"
        return F.dot(sigma)

    def compute_similarity(self,q1,q2):
        q1 = q1.split(" ")
        q2 = q2.split(" ")
        q1_sum = np.array([0]*self.embedded_matrix.shape[1])
        q2_sum = np.array([0]*self.embedded_matrix.shape[1])
        for word in q1:
            q1_sum += self.embedded_matrix[self.word_to_num[word],:]
        for word in q2:
            q2_sum += self.embedded_matrix[self.word_to_num[word],:]

        x1 = q1_sum/len(q1)
        x2 = q2_sum/len(q2)

        return x1.dot(x2.transpose())/(np.linalg.norm(x1) * np.linalg.norm(x2))

    def compute_accuracy(self,thresh):
        lines = ""
        numCorrectlyGuessed = 0
        numPairs = 0
        #(scores, answers) = self.compute_all_scores()
        for i in range(0,len(self.scores)):
            if sign(self.scores[i] - thresh) == int(self.answers[i]):
                numCorrectlyGuessed += 1
            numPairs += 1
        return numCorrectlyGuessed/float(numPairs)

    def compute_all_scores(self):
        top = 0
        similar_score_list = [None]
        correct_answer_list = [None]
        with open(self.data_set,'rb') as f:
            lines = list(csv.reader(f))
            similar_score_list = [None]*len(lines)
            correct_answer_list = [None]* len(lines)
            for i,line in enumerate(lines):
                q1 = preprocess(line[3])
                q2 = preprocess(line[4])
                similar_score_list[i] = self.compute_similarity(q1, q2)
                correct_answer_list[i] = line[-1]
            return similar_score_list, correct_answer_list


def sign(num):
    if num > 0:
        return 1
    return 0

if __name__ == "__main__":
    filename = sys.argv[1]
    test = QuoraWord2Vec(filename)
    threshs = np.arange(.8,1.01,.01)
    for x in threshs:
        print "thresh: %s  yields %s" % (x, test.compute_accuracy(x))
