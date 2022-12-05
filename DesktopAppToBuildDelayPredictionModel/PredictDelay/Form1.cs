using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace PredictDelay
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button2_Click(object sender, EventArgs e)
        {

        }

        private void btnCreateModel_Click(object sender, EventArgs e)
        {
            string script_exe1 = pythonLocation.Text;
            string script_exe = script_exe1.Replace(@"/", @"//")+"//python.exe";
            string csv_file = '"' + dataset.Text.Replace(@"/", @"//") + '"';
            string script1 = "-u " + '"' + script.Text.Replace(@"/", @"//") + '"';
            string target = txtTargetFeature.Text;
            string cardinality=txtCardinality.Text;
            string corrcoeff=txtCoeffMatrix.Text;
            string varThresh=txtVarianceThreshold.Text;
            
            
            string args = string.Format(
                "{0} {1} {2} {3} {4} {5}",
                script1, csv_file, target, cardinality, corrcoeff, varThresh);;

            string[] resultList = executeScript1(script_exe,args, txtStatus);

            try
            {
                txtAlgUsed.Text=resultList[0];
                txtRecall.Text=resultList[1];
                txtAccuracy.Text=resultList[2];
                txtF1.Text=resultList[3];
                txtPrecision.Text=resultList[4];
                txtROC.Text=resultList[5];
            }
            catch (Exception ex)
            { MessageBox.Show(ex.Message); }


        }

        string[] executeScript1(string a,string b,TextBox c)
        {
            StreamWriter swerr = new StreamWriter("error.txt");
            StreamWriter swout = new StreamWriter("output.txt");
            txtStatus.Text = "Starting...";
            var countline = 0;
            string[] result = new string[6];
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = a,
                    Arguments = b,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                },
                EnableRaisingEvents = true
            };
            txtStatus.Text = "Preparing Data, please wait!";           
            //process.ErrorDataReceived += new DataReceivedEventHandler((sender1, e1) =>
            //{
            //    swerr.WriteLine(e1.Data);
            //});
            process.OutputDataReceived += new DataReceivedEventHandler((sender, e) =>
            {

                swout.WriteLine(e.Data);

                if (e.Data != null)
                {
                    if (e.Data.Length >= 3)
                    {
                        if (e.Data.Substring(0, 6) == "status")
                        {
                            txtStatus.Text = e.Data.Substring(7);
                        }
                        else if (e.Data.Substring(0, 7) == "result1")
                        {
                            result[countline] = e.Data.Substring(8);
                            countline++;
                        }

                        else if (e.Data.Substring(0, 6) == "result")
                        {
                            result[countline] = e.Data.Substring(8,5);
                            countline++;
                        }

                    }
                }
            }
            );

            process.Start();
            //process.BeginErrorReadLine();
            process.BeginOutputReadLine();
            process.WaitForExit();
            swout.Close();
            swerr.Close();
            return result;
        }

       
    }
}
