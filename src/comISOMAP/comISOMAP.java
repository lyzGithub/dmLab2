package comISOMAP;

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

public class comISOMAP{
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
	
	public comISOMAP() throws FileNotFoundException{
		//init
		buildMatrix(sonarTrainPath);
		buildMatrix(sonarTestPath);
		buildMatrix(spliceTrainPath);
		buildMatrix(spliceTestPath);
		double startTime = 0;
		double endTime = 0;
		
		System.out.println("\n\n~~~~~~~~~~~~~~~~~~~~~sonar ISOMAP~~~~~~~~~~~~~~~~~~~~~");
		startTime = System.currentTimeMillis();
		sonarISOMAP();
		endTime = System.currentTimeMillis();
		System.out.println("Time spend for sonar ISOMAP: " +(endTime- startTime)+" ms.");
		
		System.out.println("\n\n~~~~~~~~~~~~~~~~~~~~~splice ISOMAP~~~~~~~~~~~~~~~~~~~~~");
		startTime = System.currentTimeMillis();
		spliceISOMAP();
		endTime = System.currentTimeMillis();
		System.out.println("Time spend for splice ISOMAP: " +(endTime- startTime)+" ms.");
		
	}
	private String[]  predictLabelISOMAPMix(int allLineMix,int trainLine,int testLine,Array2DRowRealMatrix Xk,String[] trainLabel){
		String [] predictLabel = new String[testLine];
		for(int i = trainLine; i<allLineMix; i++){
			double []vectorTest = Xk.getRow(i);
			String label = "";
			double minDistance = Double.MAX_VALUE;
			for(int j = 0; j<trainLine; j++){
				double []vectorTrain = Xk.getRow(j);
				double disTemp = disVector(vectorTest,vectorTrain);
				//System.out.println("disTemp:"+disTemp + " minDistance:" + minDistance);
				if(disTemp<minDistance){
					minDistance = disTemp;
					label = trainLabel[j];
				}
			}
			predictLabel[i-trainLine] = label;
			//System.out.println(label);
		}
		return predictLabel;
	}
	private void sonarISOMAP(){
		Array2DRowRealMatrix matrixSonarTrain = sonarTrainMatrix;
		Array2DRowRealMatrix matrixSonarTest = sonarTestMatrix;
		int allLineMix = allLineSonarTrain + allLineSonarTest;
		Array2DRowRealMatrix matrixMix = new Array2DRowRealMatrix(new double[allLineMix][dimSonarTrain]);
		for(int i =0; i<allLineMix; i++){
			if(i<allLineSonarTrain){
				//System.out.println(matrixSonarTrain.getRowVector(i));
				matrixMix.setRow(i, matrixSonarTrain.getRow(i));
			}
			else{
				//System.out.println(matrixSonarTest.getRowVector(i-allLineSonarTrain));
				matrixMix.setRow(i, matrixSonarTest.getRow(i-allLineSonarTrain));
			}
		}
		
		
		
		//build weighted graph
		int k = 6;
		HashMap<Integer,List<double[]>> weigtedMixGraph = returnISOMAPKnnWeightedGraph(k, matrixMix,allLineMix);
		
		for(int km=10; km<=30; km+=10){
			Array2DRowRealMatrix XkMix = comISOMAPXnm(weigtedMixGraph,k,km,allLineMix);
			
			//System.out.println("XkMix: "+XkMix);
			String []predicLabel = predictLabelISOMAPMix(allLineMix,allLineSonarTrain,allLineSonarTest,XkMix,sonarTrainLabel);
			
			int hitNum = 0;
			for(int i=0; i<allLineSonarTest; i++){
				//System.out.println(labelSplicePredict[i] +" "+ SpliceTestLabel[i]);
				if(predicLabel[i].equals(sonarTestLabel[i]) )
					hitNum ++;
			}
			System.out.print("km's num: " +km );
			System.out.println( ", allLineSonarTest:"+ allLineSonarTest+", hitNum: "+hitNum 
					+ ", hitRate:  "+(double)(hitNum)/(double)allLineSonarTest);
		}
		/*
		System.out.println("split two matrix!!! sonar");
		HashMap<Integer,List<double[]>> weigtedTrainGraph = returnISOMAPKnnWeightedGraph(k, matrixSonarTrain,allLineSonarTrain);
		HashMap<Integer,List<double[]>> weigtedTestGraph = returnISOMAPKnnWeightedGraph(k, matrixSonarTest,allLineSonarTest);
		for(int km=10; km<=30; km+=10){
			//System.out.println("km:"+km);
			Array2DRowRealMatrix XkTrain = comISOMAPXnm(weigtedTrainGraph,k,km,allLineSonarTrain);
			Array2DRowRealMatrix XkTest = comISOMAPXnm(weigtedTestGraph,k,km,allLineSonarTest);	
			String []predicLabel = predictLabelISOMAP(XkTest,XkTrain,allLineSonarTest,allLineSonarTrain,sonarTrainLabel);
			int hitNum = 0;
			for(int i=0; i<allLineSonarTest; i++){
				//System.out.println(labelSplicePredict[i] +" "+ SpliceTestLabel[i]);
				if(predicLabel[i].equals(sonarTestLabel[i]) )
					hitNum ++;
			}
			System.out.print("km's num: " +km );
			System.out.println( ", allLineSonarTest:"+ allLineSonarTest+", hitNum: "+hitNum 
					+ ", hitRate:  "+(double)(hitNum)/(double)allLineSonarTest);
		}
		*/
	}
	
	private void spliceISOMAP(){
		Array2DRowRealMatrix matrixTrain = spliceTrainMatrix;
		Array2DRowRealMatrix matrixTest = spliceTestMatrix;
		//build weighted graph
		int allLineMix = allLineSpliceTrain+allLineSpliceTest;
		Array2DRowRealMatrix matrixMix = new Array2DRowRealMatrix(new double[allLineMix][dimSpliceTrain]);
		for(int i =0; i<allLineMix; i++){
			if(i<allLineSpliceTrain)
				matrixMix.setRow(i, matrixTrain.getRow(i));
			else
				matrixMix.setRow(i, matrixTest.getRow(i-allLineSpliceTrain));
		}
		//build weighted graph
		int k = 6;
		HashMap<Integer,List<double[]>> weigtedMixGraph = returnISOMAPKnnWeightedGraph(k, matrixMix,allLineMix);
		for(int km=10; km<=10; km+=10){
			Array2DRowRealMatrix XkMix = comISOMAPXnm(weigtedMixGraph,k,km,allLineMix);
			//System.out.println("XkMix: "+XkMix);
			String []predicLabel = predictLabelISOMAPMix(allLineMix,allLineSpliceTrain,allLineSpliceTest,XkMix,spliceTrainLabel);
			int hitNum = 0;
			for(int i=0; i<allLineSpliceTest; i++){
				//System.out.println(labelSplicePredict[i] +" "+ SpliceTestLabel[i]);
				if(predicLabel[i].equals(spliceTestLabel[i]) )
					hitNum ++;
			}
			System.out.print("km's num: " +km );
			System.out.println( ", allLineSpliceTest:"+ allLineSpliceTest+", hitNum: "+hitNum 
					+ ", hitRate:  "+(double)(hitNum)/(double)allLineSpliceTest);
		}
		
		/*
		HashMap<Integer,List<double[]>> weigtedTrainGraph = returnISOMAPKnnWeightedGraph(k, matrixSonarTrain,allLineSpliceTrain);
		HashMap<Integer,List<double[]>> weigtedTestGraph = returnISOMAPKnnWeightedGraph(k, matrixSonarTest,allLineSpliceTest);
		for(int km=10; km<=30; km+=10){
			//System.out.println("km:"+km);
			//System.out.println("compute weight XkTrain!");
			Array2DRowRealMatrix XkTrain = comISOMAPXnm(weigtedTrainGraph,k,km,allLineSpliceTrain);
			//System.out.println("compute weight XkTest!");
			Array2DRowRealMatrix XkTest = comISOMAPXnm(weigtedTestGraph,k,km,allLineSpliceTest);
			String []predicLabel = predictLabelISOMAP(XkTest,XkTrain,allLineSpliceTest,allLineSpliceTrain,spliceTrainLabel);
			int hitNum = 0;
			for(int i=0; i<allLineSpliceTest; i++){
				//System.out.println(labelSplicePredict[i] +" "+ SpliceTestLabel[i]);
				if(predicLabel[i].equals(spliceTestLabel[i]) )
					hitNum ++;
			}
			System.out.print("k's num: " +k );
			System.out.println( ", allLineSpliceTest:"+ allLineSpliceTest+", hitNum: "+hitNum 
					+ ", hitRate:  "+(double)(hitNum)/(double)allLineSpliceTest);
		}
		*/
	}
	/*private String[] predictLabelISOMAP(Array2DRowRealMatrix testM, Array2DRowRealMatrix trainM,int allLineTest,int allLineTrain,String labelTrain[]){
		
		String []label = new String[allLineTest];
		for(int i=0; i<allLineTest; i++){
			double []vectorTest = testM.getRow(i);
			String predictLabel = "";
			double minDistance = Double.MAX_VALUE;
			for(int j = 0; j<allLineTrain; j++){
				double []vectorTrain = trainM.getRow(j);
				double distance = disVector(vectorTest,vectorTrain);
				
				if(distance < minDistance){
					predictLabel =  labelTrain[j];
					//System.out.println("exchange label in " + predictLabel);
					minDistance = distance;
				}
			}
			label[i] = predictLabel;
		}
		
		return label;
	}*/
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	private Array2DRowRealMatrix comISOMAPXnm(HashMap<Integer,List<double[]>> weigtedGraph,int k,int km,int allLine){
		//System.out.println("in comISOMAPXnm! " );
		SimpleWeightedGraph simWG = new SimpleWeightedGraph(DefaultWeightedEdge.class);//<Integer,DefaultWeightedEdge>
		//System.out.println("build graph simWG!");
		Iterator iter = weigtedGraph.entrySet().iterator();
		while (iter.hasNext()) {
			HashMap.Entry entry = (HashMap.Entry) iter.next();
			int key = (Integer)entry.getKey();
			//System.out.println("key: "+ key);
			simWG.addVertex(key);
			List<double[]> val = (List<double[]>)entry.getValue();
			for(int i=0; i<k; i++){
				double []temp = val.get(i);
				if(false == simWG.containsVertex((int)temp[0]) )
					simWG.addVertex((int)temp[0]);
				if(false == simWG.containsEdge(key, (int)temp[0])){
					simWG.addEdge(key, (int)temp[0]);
					simWG.setEdgeWeight(simWG.getEdge(key, (int)temp[0]), temp[1]);
					//System.out.println( " to " + temp[0] + " dis: " + temp[1]);
				}
			}
		}
		System.out.println("build graph fwSP!");
		FloydWarshallShortestPaths fwSP = new FloydWarshallShortestPaths(simWG);
		
		System.out.println("build graph disMatrixAll2!");
		RealMatrix disMatrixAll2 = new Array2DRowRealMatrix(new double[allLine][allLine]);
		for(int i=0; i<allLine; i++){
			for(int j=0; j<allLine; j++){
				if(i == j){
					disMatrixAll2.addToEntry(i, j, 0);
					continue;
				}
				//System.out.println("find path dis!");
				double disValue = fwSP.shortestDistance(i, j);
				/*DijkstraShortestPath dijP = new DijkstraShortestPath(simWG, i,j);
				double disValueDis = dijP.getPathLength();
				if(disValue != disValueDis)
					System.out.println("path length is not equal!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"+i+" "+j+ " "+disValue+" "+disValueDis);
					//dis is little difference 
				*/
				if(Double.isInfinite(disValue)){
					System.out.println("Warning, do not connect!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
					System.exit(0);
				}
				disMatrixAll2.addToEntry(i, j, disValue*disValue);
			}
			//System.out.println("vertex " + i);
		}
		//System.out.println("disMatrixAll2 "+ disMatrixAll2);
		//System.out.println("build graph tempJ!");
		RealMatrix tempJ = new Array2DRowRealMatrix(new double[allLine][allLine]);
		for(int i=0; i<allLine; i++){
			for(int j=0; j<allLine; j++){
				if(i == j){
					tempJ.setEntry(i, j, 1 - 1/(double)allLine);
					continue;
				}
				tempJ.setEntry(i, j, -1/(double)allLine);
					
			}
		}
		/*for(int i=0; i<allLine; i++){
			System.out.println(i+ " "+ tempJ.getRowVector(i));
		}*/
		//System.out.println("build graph tempB!");
		RealMatrix tmp1 = tempJ.multiply(disMatrixAll2);
		RealMatrix tmp2 = tmp1.multiply(tempJ);
		RealMatrix tempB  = tmp2.scalarMultiply(-0.5);
		//System.out.println("tempB: "+ tempB);
		System.out.println("eigen dec!");
		/*
		 * The eigen decomposition of matrix A is a set of two matrices:
		 *  V and D such that A = V ¡Á D ¡Á VT. A, V and D are all m ¡Á m matrices.
		 */
		EigenDecomposition solverEigen = new EigenDecomposition(tempB);//V:eigenvector matrix, D:eigen values;
		System.out.println("eigen dec finish!");
		/*for(int i=0; i<eigenValues.length; i++){
			System.out.println("eigenValues :"+eigenValues[i]);
		}*/
		
		Array2DRowRealMatrix Enm = new Array2DRowRealMatrix(new double[allLine][km]);
		//Array2DRowRealMatrix Amm = new Array2DRowRealMatrix(new double[km][km]);
		Array2DRowRealMatrix Amm5 = new Array2DRowRealMatrix(new double[km][km]);
		for(int ki=0; ki<km; ki++){
			Enm.setColumnVector(ki, solverEigen.getEigenvector(ki));
			double eigenValueTmp = solverEigen.getRealEigenvalue(ki);
			if(eigenValueTmp < 0){
				//System.out.println("eigenValues[ki] is <0.");
				System.exit(-1);
			}
			//System.out.println("eigenValues[ki]:" + Math.sqrt(eigenValueTmp));
			Amm5.setEntry(ki, ki, Math.sqrt(eigenValueTmp));//
		}
		
		//System.out.println("Enm: "+ Enm);
		//System.out.println("Enm5: "+ Amm5);
		Array2DRowRealMatrix Xkm = Enm.multiply(Amm5);
		///System.out.println("Xkm: "+ Xkm);
		return Xkm;
	}
	
	private HashMap<Integer,List<double[]>> returnISOMAPKnnWeightedGraph(int k, Array2DRowRealMatrix matrix,int allLine){
		HashMap<Integer,List<double[]>> weigtedGraph = new HashMap<Integer,List<double[]>>();
		for(int i=0; i<allLine; i++){
			List<double[]> minDis = new ArrayList<double[]>();
			for(int m =0; m<k; m++){
				double []z = new double[2];
				z[1] = Double.MAX_VALUE;
				minDis.add(z);
			}
			for(int j = 0; (j<allLine) ; j++){
				if(i==j){
					continue;
				}
				double tempDis = disVector(matrix.getRow(i),matrix.getRow(j));
				double []relace = new double[2];
				relace[0] = j;
				relace[1] = tempDis;
				for(int m =k-1; m>=0; m--){
					if(relace[1]< (minDis.get(m))[1] ){
						double []tmt = minDis.get(m);
						minDis.set(m, relace);
						relace = tmt;
					}
				}
			}
			
			/*System.out.println("row "+i+"'s min: ");
			for(int index =0; index<k; index ++){
				System.out.println(" to "+(int)minDis.get(index)[0]+", dis"+minDis.get(index)[1]);
			}*/
			weigtedGraph.put(i, minDis);
		}
		
		
		return weigtedGraph;
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