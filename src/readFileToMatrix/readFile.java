package readFileToMatrix;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

public class readFile{
	private String sonarTestPath = "data/data1/sonar-test.txt";
	private String sonarTrainPath = "data/data1/sonar-train.txt";
	private String spliceTestPath = "data/data2/splice-test.txt";
	private String spliceTrainPath = "data/data2/splice-train.txt";
	private Array2DRowRealMatrix sonarTrainMatrix;
	private Array2DRowRealMatrix sonarTestMatrix;
	private Array2DRowRealMatrix spliceTestMatrix;
	private Array2DRowRealMatrix spliceTrainMatrix;
	
	private String []sonarTrainLabel;
	private String []sonarTestLabel;
	private String []spliceTestLabel;
	private String []spliceTrainLabel;
	
	private int allLineSonarTrain;
	private int allLineSonarTest;
	private int allLineSpliceTrain;
	private int allLineSpliceTest;
	
	private int dimSonarTrain;
	private int dimSonarTest;
	private int dimSpliceTrain;
	private int dimSpliceTest;
	
	public readFile() throws FileNotFoundException{
		//init
		buildMatrix(sonarTrainPath);
		buildMatrix(sonarTestPath);
		buildMatrix(spliceTrainPath);
		buildMatrix(spliceTestPath);
		//System.out.println(sonarTrainMatrix);
		System.out.println("~~~~~~~~~~~~~~~~~~~~~sonar PCA~~~~~~~~~~~~~~~~~~~~~");
		//sonarPCA();
		System.out.println("~~~~~~~~~~~~~~~~~~~~~splice PCA~~~~~~~~~~~~~~~~~~~~~");
		//splicePCA();
		System.out.println("~~~~~~~~~~~~~~~~~~~~~sonar SVD~~~~~~~~~~~~~~~~~~~~~");
		//sonarSVD();
		System.out.println("~~~~~~~~~~~~~~~~~~~~~splice SVD~~~~~~~~~~~~~~~~~~~~~");
		//spliceSVD();
		System.out.println("~~~~~~~~~~~~~~~~~~~~~sonar ISOMAP~~~~~~~~~~~~~~~~~~~~~");
		sonarISOMAP();
		System.out.println("~~~~~~~~~~~~~~~~~~~~~splice ISOMAP~~~~~~~~~~~~~~~~~~~~~");
		spliceISOMAP();
		
	}
	
	private void sonarISOMAP(){
		Array2DRowRealMatrix matrixSonarTrain = sonarTrainMatrix;
		//build weighted graph
		int k = 4;
		HashMap<Integer,List<double[]>> weigtedTrainGraph = returnKnnWeightedGraph(k, matrixSonarTrain);
		Array2DRowRealMatrix disMatrixAll = new Array2DRowRealMatrix(new double[allLineSonarTrain][allLineSonarTrain]);
		
		
		//System.out.println(weigtedTrainGraph);
		Array2DRowRealMatrix matrixSonarTest = sonarTestMatrix;
		
	}
	private HashMap<Integer,List<double[]>> returnKnnWeightedGraph(int k, Array2DRowRealMatrix matrix){
		HashMap<Integer,List<double[]>> weigtedGraph = new HashMap<Integer,List<double[]>>();
		for(int i=0; i<allLineSonarTrain; i++){
			List<double[]> minDis = new ArrayList<double[]>();
			for(int m =0; m<k; m++){
				double []z = new double[2];
				z[1] = Double.MAX_VALUE;
				minDis.add(z);
			}
			for(int j = 0; j<allLineSonarTrain && j != i; j++){
				double tempDis = disVector(matrix.getRow(i),matrix.getRow(j));
				//System.out.println("tempDis:"+tempDis);
				for(int m =k-1; m>=0; m--){
					
					if(tempDis< (minDis.get(m))[1] ){
						double []newOne = new double[2];
						newOne[0] = j;
						newOne[1] = tempDis;
						//System.out.println("tempDis:"+tempDis);
						minDis.set(m, newOne);
						break;
					}
					
				}
			}
			/*
			System.out.println("row "+i+"'s min: ");
			for(int index =0; index<k; index ++){
				System.out.println(" to "+(int)minDis.get(index)[0]+", dis"+minDis.get(index)[1]);
			}*/
			weigtedGraph.put(i, minDis);
		}
		
		
		return weigtedGraph;
	}
	
	private void spliceISOMAP(){
		
	}
	//return dis of twoo vector in double[]
	private double disVector(double []v1,double []v2){
		
		double dis =0;
		
		if(v1.length != v2.length){
			System.out.println("in disVector length is not the same!");
			System.exit(0);
		}
		for(int i=0; i<v1.length; i++){
			dis += (v1[i]-v2[i])*(v1[i]-v2[i]);
		}
		return Math.sqrt(dis);
	}
	private void sonarSVD(){
		Array2DRowRealMatrix matrixATrain = sonarTrainMatrix;
		Array2DRowRealMatrix matrixATest = sonarTestMatrix;
		
		SingularValueDecomposition svdSolver = new SingularValueDecomposition(matrixATrain);
		//double []singularValue = svdSolver.getSingularValues();
		for(int k=10; k<=30; k+=10){
			//System.out.print("k num:" + k);
			//RealMatrix U =  svdSolver.getU();
			/*
			 * The Singular Value Decomposition of matrix A is a set of three matrices: U, Σ and V such that A = U × Σ × VT.
			 *  Let A be a m × n matrix, then U is a m × p orthogonal matrix, 
			 *  Σ is a p × p diagonal matrix with positive or null elements, 
			 *  V is a p × n orthogonal matrix (hence VT is also orthogonal) where p=min(m,n).
			 */
			RealMatrix V =  svdSolver.getV();
			System.out.println(V.getColumnDimension()+"  "+V.getRowDimension());
			//System.out.println(U.getColumnDimension());
			Array2DRowRealMatrix Uk = new Array2DRowRealMatrix(new double[dimSonarTrain][k]);
			for(int i=0; i<k; i++){
				Uk.setColumn(i, V.getRow(i) );
			}
			//Array2DRowRealMatrix UkT = (Array2DRowRealMatrix) Uk.transpose();
			Array2DRowRealMatrix newCorMatrixTrain = matrixATrain.multiply(Uk);
			Array2DRowRealMatrix newCorMatrixTest =  matrixATest.multiply(Uk);
			String []labelPrdict = predictSVDSonar(newCorMatrixTest,newCorMatrixTrain);
			int hitNum = 0;
			
			for(int i=0; i<allLineSonarTest; i++){
				//System.out.println(labelSplicePredict[i] +" "+ SpliceTestLabel[i]);
				if(labelPrdict[i].equals(sonarTestLabel[i]) )
					hitNum ++;
			}
			System.out.print("k's num: " +k );
			System.out.println( ", allLineSonarTest:"+ allLineSonarTest+", hitNum: "+hitNum 
					+ ", hitRate:  "+(double)(hitNum)/(double)allLineSonarTest);
		
		}
	}
	private String[] predictSVDSonar(Array2DRowRealMatrix testM, Array2DRowRealMatrix trainM){
		String []label = new String[allLineSonarTest];
		for(int i=0; i<allLineSonarTest; i++){
			double []vectorTest = testM.getRow(i);
			String predictLabel = "";
			double minDistance = Double.MAX_VALUE;
			for(int j = 0; j<allLineSonarTrain; j++){
				double []vectorTrain = trainM.getRow(j);
				double distance = 0;
				for(int k=0; k<vectorTrain.length; k++){
					distance += (vectorTest[k] - vectorTrain[k]) * (vectorTest[k] - vectorTrain[k]);
				}
				if(distance < minDistance){
					predictLabel =  sonarTrainLabel[j];
					//System.out.println("exchange label in " + predictLabel);
					minDistance = distance;
				}
				
			}
			label[i] = predictLabel;
		}
		
		return label;
	}
	
	private void spliceSVD(){
		Array2DRowRealMatrix matrixATrain = spliceTrainMatrix;
		Array2DRowRealMatrix matrixATest = spliceTestMatrix;
		
		SingularValueDecomposition svdSolver = new SingularValueDecomposition(matrixATrain);
		//double []singularValue = svdSolver.getSingularValues();
		for(int k=10; k<=30; k+=10){
			//System.out.print("k num:" + k);
			//RealMatrix U =  svdSolver.getU();
			/*
			 * The Singular Value Decomposition of matrix A is a set of three matrices: U, Σ and V such that A = U × Σ × VT.
			 *  Let A be a m × n matrix, then U is a m × p orthogonal matrix, 
			 *  Σ is a p × p diagonal matrix with positive or null elements, 
			 *  V is a p × n orthogonal matrix (hence VT is also orthogonal) where p=min(m,n).
			 */
			RealMatrix V =  svdSolver.getV();
			System.out.println(V.getColumnDimension()+"  "+V.getRowDimension());
			//System.out.println(U.getColumnDimension());
			Array2DRowRealMatrix Uk = new Array2DRowRealMatrix(new double[dimSpliceTrain][k]);
			for(int i=0; i<k; i++){
				Uk.setColumn(i, V.getRow(i) );
			}
			//Array2DRowRealMatrix UkT = (Array2DRowRealMatrix) Uk.transpose();
			Array2DRowRealMatrix newCorMatrixTrain = matrixATrain.multiply(Uk);
			Array2DRowRealMatrix newCorMatrixTest =  matrixATest.multiply(Uk);
			String []labelPrdict = predictSVDSplice(newCorMatrixTest,newCorMatrixTrain);
			int hitNum = 0;
			
			for(int i=0; i<allLineSpliceTest; i++){
				//System.out.println(labelSplicePredict[i] +" "+ SpliceTestLabel[i]);
				if(labelPrdict[i].equals(spliceTestLabel[i]) )
					hitNum ++;
			}
			System.out.print("k's num: " +k );
			System.out.println( ", allLineSpliceTest:"+ allLineSpliceTest+", hitNum: "+hitNum 
					+ ", hitRate:  "+(double)(hitNum)/(double)allLineSpliceTest);
		
		}
	}
	private String[] predictSVDSplice(Array2DRowRealMatrix testM, Array2DRowRealMatrix trainM){
		String []label = new String[allLineSpliceTest];
		for(int i=0; i<allLineSpliceTest; i++){
			double []vectorTest = testM.getRow(i);
			String predictLabel = "";
			double minDistance = Double.MAX_VALUE;
			for(int j = 0; j<allLineSpliceTrain; j++){
				double []vectorTrain = trainM.getRow(j);
				double distance = 0;
				for(int k=0; k<vectorTrain.length; k++){
					distance += (vectorTest[k] - vectorTrain[k]) * (vectorTest[k] - vectorTrain[k]);
				}
				if(distance < minDistance){
					predictLabel =  spliceTrainLabel[j];
					//System.out.println("exchange label in " + predictLabel);
					minDistance = distance;
				}
				
			}
			label[i] = predictLabel;
		}
		
		return label;
	}
	
	
	private void splicePCA(){
		Array2DRowRealMatrix matrixATrain = spliceTrainMatrix;
		Array2DRowRealMatrix matrixB = matrixATrain;
		//PCA first step: cov matrix
		//each row mean
		//System.out.println(dimSpliceTrain+matrixB.getColumnDimension());
		for(int i=0; i<dimSpliceTrain; i++){
			 double []v = matrixB.getColumn(i);
			 double mean = 0;
			 double allSumColumn = 0;
			 int lengthOfV = v.length;
			 for(int j =0; j<lengthOfV;j++){
				 allSumColumn += v[j];
			 }
			 mean = allSumColumn/lengthOfV;
			 for(int j =0; j<lengthOfV;j++){
				 v[j] -= mean;
			 }
			 matrixB.setColumn(i, v);
			 
		}
		//System.out.println("matrixATrain:"+matrixATrain);
		//System.out.println("matrixB:"+matrixB);
		
		Array2DRowRealMatrix matrixBT;
		
		matrixBT = (Array2DRowRealMatrix) matrixB.transpose();
		//System.out.println("matrixBT:"+matrixBT);
		//matrixZ is A's cov matrix
		double n1 = (double)1/(double)(allLineSpliceTrain-1);
		//System.out.println("!!!!!!!!!!"+n1);
		Array2DRowRealMatrix matrixZ = 
				(Array2DRowRealMatrix) (matrixBT.multiply(matrixB)).scalarMultiply(n1);
		//System.out.println("rowZ:"+matrixZ.getColumnDimension());
		EigenDecomposition solver = new EigenDecomposition(matrixZ);
		//double []eigenValues = solver.getRealEigenvalues();//特征值 rang from max to min

		/*for(int i=0; i<eigenValues.length; i++){
			System.out.println("eigenValues:"+eigenValues[i]);
		}*/
		
		//根据K选取特征向量组成矩阵N（n*k）
		for(int k1 = 10; k1 <= 30; k1 +=10 ){
			Array2DRowRealMatrix k1MK = new  Array2DRowRealMatrix(new double[dimSpliceTrain][k1]);
			Array2DRowRealMatrix eigenV = (Array2DRowRealMatrix) solver.getV();
			for(int i = 0; i<k1; i++){
				double []vector = eigenV.getColumn(i);
				k1MK.setColumn(i, vector);
			}
		
			//compute the PC matrix N
			Array2DRowRealMatrix k1CTrain = matrixATrain.multiply(k1MK);
			Array2DRowRealMatrix matrixATest = spliceTestMatrix;
			Array2DRowRealMatrix k1CTest = meanComCentry(matrixATest.multiply(k1MK));
			//System.out.println(k1CTest);
		
			//apply 1-NN:find the most min distance of k1CTrain
			String []labelSplicePredict = predictPCALabelSplice(k1CTest,k1CTrain);
			int hitNum = 0;
			
			for(int i=0; i<allLineSpliceTest; i++){
				//System.out.println(labelSplicePredict[i] +" "+ SpliceTestLabel[i]);
				if(labelSplicePredict[i].equals(spliceTestLabel[i]) )
					hitNum ++;
			}
			System.out.print("k's num: " +k1 );
			System.out.println( ", allLineSpliceTest:"+ allLineSpliceTest+", hitNum: "+hitNum + ", hitRate:  "+(double)(hitNum)/(double)allLineSpliceTest);
		}
	}
	//apply 1-NN:find the most min distance of k1CTrain
		private String[] predictPCALabelSplice(Array2DRowRealMatrix testM, Array2DRowRealMatrix trainM){
			String []label = new String[allLineSpliceTest];
			for(int i=0; i<allLineSpliceTest; i++){
				double []vectorTest = testM.getRow(i);
				String predictLabel = "";
				double minDistance = Double.MAX_VALUE;
				for(int j = 0; j<allLineSpliceTrain; j++){
					double []vectorTrain = trainM.getRow(j);
					double distance = 0;
					for(int k=0; k<vectorTrain.length; k++){
						distance += (vectorTest[k] - vectorTrain[k]) * (vectorTest[k] - vectorTrain[k]);
					}
					if(distance < minDistance){
						predictLabel =  spliceTrainLabel[j];
						//System.out.println("exchange label in " + predictLabel);
						minDistance = distance;
					}
					
				}
				label[i] = predictLabel;
			}
			
			return label;
		}
		
	/*************************************************************************/
	private void sonarPCA() throws FileNotFoundException{
		
		Array2DRowRealMatrix matrixATrain = sonarTrainMatrix;
		Array2DRowRealMatrix matrixB = matrixATrain;
		//PCA first step: cov matrix
		//each row mean
		//System.out.println(dimSonarTrain+matrixB.getColumnDimension());
		for(int i=0; i<dimSonarTrain; i++){
			 double []v = matrixB.getColumn(i);
			 double mean = 0;
			 double allSumColumn = 0;
			 int lengthOfV = v.length;
			 for(int j =0; j<lengthOfV;j++){
				 allSumColumn += v[j];
			 }
			 mean = allSumColumn/lengthOfV;
			 for(int j =0; j<lengthOfV;j++){
				 v[j] -= mean;
			 }
			 matrixB.setColumn(i, v);
			 
		}
		//System.out.println("matrixATrain:"+matrixATrain);
		//System.out.println("matrixB:"+matrixB);
		
		Array2DRowRealMatrix matrixBT;
		
		matrixBT = (Array2DRowRealMatrix) matrixB.transpose();
		//System.out.println("matrixBT:"+matrixBT);
		//matrixZ is A's cov matrix
		double n1 = (double)1/(double)(allLineSonarTrain-1);
		//System.out.println("!!!!!!!!!!"+n1);
		Array2DRowRealMatrix matrixZ = 
				(Array2DRowRealMatrix) (matrixBT.multiply(matrixB)).scalarMultiply(n1);
		//System.out.println("rowZ:"+matrixZ.getColumnDimension());
		EigenDecomposition solver = new EigenDecomposition(matrixZ);
		//double []eigenValues = solver.getRealEigenvalues();//特征值 rang from max to min

		/*for(int i=0; i<eigenValues.length; i++){
			System.out.println("eigenValues:"+eigenValues[i]);
		}*/
		
		//根据K选取特征向量组成矩阵N（n*k）
		for(int k1 = 10; k1 <= 30; k1 +=10 ){
			
			Array2DRowRealMatrix k1MK = new  Array2DRowRealMatrix(new double[dimSonarTrain][k1]);
			//Array2DRowRealMatrix eigenV = (Array2DRowRealMatrix) solver.getV();
			for(int i = 0; i<k1; i++){
				//double []vector = eigenV.getColumn(i);
				RealVector rev = solver.getEigenvector(i);
				//System.out.println(rev);
				k1MK.setColumnVector(i, rev);
			}
			//System.out.println(k1MK.getColumnDimension()+" "+k1MK.getRowDimension());
			//compute the PC matrix N
			Array2DRowRealMatrix k1CTrain = matrixATrain.multiply(k1MK);
			Array2DRowRealMatrix matrixATest = sonarTestMatrix;
			Array2DRowRealMatrix k1CTest = meanComCentry(matrixATest.multiply(k1MK));
			//System.out.println(k1CTest);
			/*for(int i =0; i<k1CTest.getRowDimension(); i++){
				System.out.println(k1CTest.getRowVector(i));
			}*/
			//apply 1-NN:find the most min distance of k1CTrain
			
			String []labelSonarPredict = predictPCALabelSonar(k1CTest,k1CTrain);
			int hitNum = 0;
			for(int i=0; i<allLineSonarTest; i++){
				//System.out.println(labelSonarPredict[i] +" "+ sonarTestLabel[i]);
				if(labelSonarPredict[i].equals(sonarTestLabel[i]) )
					hitNum ++;
			}
			System.out.print("k's num: " +k1 );
			System.out.println(", allLineSonarTest: "+ allLineSonarTest +", hitNum: "+ hitNum + ", hitRate:  "+(double)(hitNum)/(double)allLineSonarTest);
		}
	}
	
	//apply 1-NN:find the most min distance of k1CTrain
	private String[] predictPCALabelSonar(Array2DRowRealMatrix testM, Array2DRowRealMatrix trainM){
		String []label = new String[allLineSonarTest];
		for(int i=0; i<allLineSonarTest; i++){
			double []vectorTest = testM.getRow(i);
			String predictLabel = "";
			double minDistance = Double.MAX_VALUE;
			for(int j = 0; j<allLineSonarTrain; j++){
				double []vectorTrain = trainM.getRow(j);
				double distance = 0;
				for(int k=0; k<vectorTrain.length; k++){
					distance += (vectorTest[k] - vectorTrain[k]) * (vectorTest[k] - vectorTrain[k]);
				}
				if(distance < minDistance){
					predictLabel =  sonarTrainLabel[j];
					//System.out.println("exchange label in " + predictLabel);
					minDistance = distance;
				}
				
			}
			label[i] = predictLabel;
		}
		
		return label;
	}
	
	
	
	private void buildMatrix(String path) throws FileNotFoundException{
		
		String []label;
		int allLine=0;
		int dim=0;
		////////////////////////////////////////
		File file = new File(path);
		Scanner inputNext = new Scanner(file);
		StringBuilder strb = new StringBuilder();
		while(inputNext.hasNext()){
			strb.append(inputNext.next()+"\n");
		}
		inputNext.close();
		String []lineString =  strb.toString().split("\n");
		allLine = lineString.length;
		dim = lineString[0].split(",").length-1;
		
		double [][]mtr = new double[allLine][dim];// metrix for SonarTrain
		label = new String[allLine];
		for(int i = 0; i<allLine; i++){
			String []entrys = lineString[i].split(",");
			for(int j =0; j<dim; j++){
				double temp = Double.parseDouble(entrys[j]);
				mtr[i][j] = temp;
			}
			label[i] = entrys[dim];
		}
		
		Array2DRowRealMatrix storedMatrix = new Array2DRowRealMatrix(mtr);
		if(sonarTestPath == path){
			sonarTestMatrix = storedMatrix;
			sonarTestLabel = label;
			allLineSonarTest = allLine;
			dimSonarTest = dim;
		}
		else if(sonarTrainPath == path){
			sonarTrainMatrix = storedMatrix;
			sonarTrainLabel = label;
			allLineSonarTrain = allLine;
			dimSonarTrain = dim;
		}
		else if(spliceTestPath == path){
			spliceTestMatrix = storedMatrix;
			spliceTestLabel = label;
			allLineSpliceTest = allLine;
			dimSpliceTest = dim;
		}
		else if(spliceTrainPath == path){
			spliceTrainMatrix = storedMatrix;
			spliceTrainLabel = label;
			allLineSpliceTrain = allLine;
			dimSpliceTrain = dim;
		}
		
		
		
	}
	private Array2DRowRealMatrix meanComCentry(Array2DRowRealMatrix testM){
		for(int i=0; i<testM.getColumnDimension(); i++){
			 double []v = testM.getColumn(i);
			 double mean = 0;
			 double allSumColumn = 0;
			 int lengthOfV = v.length;
			 for(int j =0; j<lengthOfV;j++){
				 allSumColumn += v[j];
			 }
			 mean = allSumColumn/lengthOfV;
			 for(int j =0; j<lengthOfV;j++){
				 v[j] -= mean;
			 }
			 testM.setColumn(i, v);
			 
		}
		return testM;
	}

	public Array2DRowRealMatrix returnSonarTrainMatrix(){
		return sonarTrainMatrix;
	}
	public void returnSonarTrainLabel(){
		
	}
	
	public Array2DRowRealMatrix returnSonarTestMatrix(){
		return sonarTestMatrix;
	}
	public void returnSonarTestLabel(){
		
	}
	
	public Array2DRowRealMatrix returnSpliceTrainMatrix(){
		return spliceTrainMatrix;
	}
	public void returnSpliceTrainLabel(){
		
	}
	
	public Array2DRowRealMatrix returnSpliceTestMatrix(){
		return spliceTestMatrix;
	}
	public void returnSpliceTestLabel(){
		
	}
}