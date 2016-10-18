package comSVD;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.jgraph.graph.DefaultEdge;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.WeightedGraph;
import org.jgrapht.alg.DijkstraShortestPath;
import org.jgrapht.alg.FloydWarshallShortestPaths;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleGraph;
import org.jgrapht.graph.SimpleWeightedGraph;

public class comSVD{
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
	
	public comSVD() throws FileNotFoundException{
		//init
		buildMatrix(sonarTrainPath);
		buildMatrix(sonarTestPath);
		buildMatrix(spliceTrainPath);
		buildMatrix(spliceTestPath);
		double startTime = 0;
		double endTime = 0;
		
		System.out.println("\n\n~~~~~~~~~~~~~~~~~~~~~sonar SVD~~~~~~~~~~~~~~~~~~~~~");
		startTime = System.currentTimeMillis();
		sonarSVD();
		endTime = System.currentTimeMillis();
		System.out.println("Time spend for sonar SVD: " +(endTime- startTime)+" ms.");
		
		System.out.println("\n\n~~~~~~~~~~~~~~~~~~~~~splice SVD~~~~~~~~~~~~~~~~~~~~~");
		startTime = System.currentTimeMillis();
		spliceSVD();
		endTime = System.currentTimeMillis();
		System.out.println("Time spend for splice SVD: " +(endTime- startTime)+" ms.");
		
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
			 * The Singular Value Decomposition of matrix A is a set of three matrices: U, ¦² and V such that A = U ¡Á ¦² ¡Á VT.
			 *  Let A be a n ¡Á m matrix, then U is a n ¡Á p orthogonal matrix, 
			 *  ¦² is a p ¡Á p diagonal matrix with positive or null elements, 
			 *  V is a p ¡Á m orthogonal matrix (hence VT is also orthogonal) where p=min(n,m).
			 */
			RealMatrix V =  svdSolver.getV();
			//System.out.println(V);
			//System.out.println(U.getColumnDimension());
			Array2DRowRealMatrix Uk = new Array2DRowRealMatrix(new double[dimSonarTrain][k]);
			for(int i=0; i<k; i++){
				Uk.setColumn(i, V.getColumn(i) );
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
				double distance = disVector(vectorTest,vectorTrain);
				
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
			 * The Singular Value Decomposition of matrix A is a set of three matrices: U, ¦² and V such that A = U ¡Á ¦² ¡Á VT.
			 *  Let A be a m ¡Á n matrix, then U is a m ¡Á p orthogonal matrix, 
			 *  ¦² is a p ¡Á p diagonal matrix with positive or null elements, 
			 *  V is a p ¡Á n orthogonal matrix (hence VT is also orthogonal) where p=min(m,n).
			 */
			RealMatrix V =  svdSolver.getV();
			//System.out.println(V.getColumnDimension()+"  "+V.getRowDimension());
			//System.out.println(U.getColumnDimension());
			Array2DRowRealMatrix Uk = new Array2DRowRealMatrix(new double[dimSpliceTrain][k]);
			for(int i=0; i<k; i++){
				Uk.setColumn(i, V.getColumn(i) );
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
	
}