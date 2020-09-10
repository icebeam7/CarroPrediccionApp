using System;
using Microsoft.ML;
using System.Linq;

namespace PrediccionCarros
{
    class Program
    {
        private static string dataset_file = @"/Users/luisbeltran/Projects/mlnet/true_car_listings.csv";
        private static string rutaModelo = @"/Users/luisbeltran/Projects/mlnet/MLModel.zip";

        static void Main(string[] args)
        {
            MLContext contexto = new MLContext();

            Console.WriteLine("Cargando datos...");
            IDataView datos = contexto.Data.LoadFromTextFile<Carro>(
                path: dataset_file,
                hasHeader: true,
                separatorChar: ',');

            var datosSplit = contexto.Data.TrainTestSplit(datos, testFraction: 0.2);
            //var datosSplit = contexto.Data.TrainTestSplit(datos, 0.2); 
            //var datosSplit = contexto.Data.TrainTestSplit(testFraction: 0.2, data: datos);

            Console.WriteLine("Preparando el entrenamiento...");
            var pipeline = contexto.Transforms
                .Categorical.OneHotEncoding(outputColumnName: "MakeEncoded", inputColumnName: "Make")
                    .Append(contexto.Transforms
                            .Categorical.OneHotEncoding(outputColumnName: "ModelEncoded",
                                        inputColumnName: "Model"))
                    .Append(contexto.Transforms
                        .Concatenate("Features", "Year", "Mileage", "MakeEncoded", "ModelEncoded"))
                    .Append(contexto.Transforms.NormalizeMinMax("Features", "Features"))
                    .AppendCacheCheckpoint(contexto)
                    ;

            var trainer = contexto.Regression.Trainers.LbfgsPoissonRegression();
            var pipelineEntrenamiento = pipeline.Append(trainer);

            Console.WriteLine("Entrenando el modelo...");
            Console.WriteLine($"Inicia: {DateTime.Now.ToShortTimeString()}");
            var modelo = pipelineEntrenamiento.Fit(datosSplit.TrainSet);
            Console.WriteLine($"Termina: {DateTime.Now.ToShortTimeString()}");

            Console.WriteLine("Haciendo predicciones sobre entrenamiento y prueba");
            IDataView prediccionesEntrenamiento = modelo.Transform(datosSplit.TrainSet);
            IDataView prediccionesPrueba = modelo.Transform(datosSplit.TestSet);

            var metricasEntrenamiento = contexto.Regression
                .Evaluate(prediccionesEntrenamiento, labelColumnName: "Label", scoreColumnName: "Score");

            var metricasPrueba = contexto.Regression
                .Evaluate(prediccionesPrueba, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine($"R-Squared Set de Entrenamiento: {metricasEntrenamiento.RSquared}");
            Console.WriteLine($"R-Squared Set de Prueba: {metricasPrueba.RSquared}");

            var crossValidation = contexto.Regression.CrossValidate(datos, pipelineEntrenamiento, numberOfFolds: 5);
            var R_Squared_Avg = crossValidation.Select(modelo => modelo.Metrics.RSquared).Average();
            Console.WriteLine($"R-Squared Cross Validation: {R_Squared_Avg}");

            Console.WriteLine("Guardando el modelo...");
            contexto.Model.Save(modelo, datos.Schema, rutaModelo);
        }
    }
}
