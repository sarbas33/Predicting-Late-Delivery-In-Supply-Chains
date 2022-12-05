namespace PredictDelay
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.btnCreateModel = new System.Windows.Forms.Button();
            this.txtStatus = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.txtVarianceThreshold = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.txtCoeffMatrix = new System.Windows.Forms.TextBox();
            this.txtCardinality = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.txtTargetFeature = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.script = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.dataset = new System.Windows.Forms.TextBox();
            this.pythonLocation = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.txtAlgUsed = new System.Windows.Forms.TextBox();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.label14 = new System.Windows.Forms.Label();
            this.txtAccuracy = new System.Windows.Forms.TextBox();
            this.txtROC = new System.Windows.Forms.TextBox();
            this.txtF1 = new System.Windows.Forms.TextBox();
            this.txtPrecision = new System.Windows.Forms.TextBox();
            this.txtRecall = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.SuspendLayout();
            // 
            // btnCreateModel
            // 
            this.btnCreateModel.Location = new System.Drawing.Point(486, 252);
            this.btnCreateModel.Name = "btnCreateModel";
            this.btnCreateModel.Size = new System.Drawing.Size(149, 86);
            this.btnCreateModel.TabIndex = 0;
            this.btnCreateModel.Text = "Create Model";
            this.btnCreateModel.UseVisualStyleBackColor = true;
            this.btnCreateModel.Click += new System.EventHandler(this.btnCreateModel_Click);
            // 
            // txtStatus
            // 
            this.txtStatus.BackColor = System.Drawing.SystemColors.Info;
            this.txtStatus.Location = new System.Drawing.Point(526, 39);
            this.txtStatus.Name = "txtStatus";
            this.txtStatus.Size = new System.Drawing.Size(280, 20);
            this.txtStatus.TabIndex = 1;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(476, 42);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(40, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "Status:";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(77, 56);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(105, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Location  of DataSet";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(56, 30);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(126, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Python .exe Parent folder";
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.groupBox2);
            this.groupBox1.Controls.Add(this.txtTargetFeature);
            this.groupBox1.Controls.Add(this.label7);
            this.groupBox1.Controls.Add(this.script);
            this.groupBox1.Controls.Add(this.label6);
            this.groupBox1.Controls.Add(this.dataset);
            this.groupBox1.Controls.Add(this.pythonLocation);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Location = new System.Drawing.Point(12, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(458, 303);
            this.groupBox1.TabIndex = 5;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Input Data Details";
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.txtVarianceThreshold);
            this.groupBox2.Controls.Add(this.label5);
            this.groupBox2.Controls.Add(this.label4);
            this.groupBox2.Controls.Add(this.txtCoeffMatrix);
            this.groupBox2.Controls.Add(this.txtCardinality);
            this.groupBox2.Controls.Add(this.label8);
            this.groupBox2.Location = new System.Drawing.Point(9, 159);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(443, 119);
            this.groupBox2.TabIndex = 16;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Optimizing Model Building";
            // 
            // txtVarianceThreshold
            // 
            this.txtVarianceThreshold.Location = new System.Drawing.Point(309, 81);
            this.txtVarianceThreshold.Name = "txtVarianceThreshold";
            this.txtVarianceThreshold.Size = new System.Drawing.Size(59, 20);
            this.txtVarianceThreshold.TabIndex = 20;
            this.txtVarianceThreshold.Text = "0.01";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(112, 84);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(191, 13);
            this.label5.TabIndex = 19;
            this.label5.Text = "Variance Threshold Method: Threshold";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(55, 56);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(248, 13);
            this.label4.TabIndex = 18;
            this.label4.Text = "Threshold for Removing Correlated Features (0.5-1)";
            // 
            // txtCoeffMatrix
            // 
            this.txtCoeffMatrix.Location = new System.Drawing.Point(309, 53);
            this.txtCoeffMatrix.Name = "txtCoeffMatrix";
            this.txtCoeffMatrix.Size = new System.Drawing.Size(59, 20);
            this.txtCoeffMatrix.TabIndex = 17;
            this.txtCoeffMatrix.Text = "0.90";
            // 
            // txtCardinality
            // 
            this.txtCardinality.Location = new System.Drawing.Point(309, 27);
            this.txtCardinality.Name = "txtCardinality";
            this.txtCardinality.Size = new System.Drawing.Size(59, 20);
            this.txtCardinality.TabIndex = 16;
            this.txtCardinality.Text = "5";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(6, 32);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(297, 13);
            this.label8.TabIndex = 15;
            this.label8.Text = "Cardinality Limit For Separating Feature Encoding Techniques";
            // 
            // txtTargetFeature
            // 
            this.txtTargetFeature.Location = new System.Drawing.Point(188, 109);
            this.txtTargetFeature.Name = "txtTargetFeature";
            this.txtTargetFeature.Size = new System.Drawing.Size(231, 20);
            this.txtTargetFeature.TabIndex = 14;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(102, 113);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(80, 13);
            this.label7.TabIndex = 13;
            this.label7.Text = "Target Feature:";
            // 
            // script
            // 
            this.script.Location = new System.Drawing.Point(188, 82);
            this.script.Name = "script";
            this.script.Size = new System.Drawing.Size(231, 20);
            this.script.TabIndex = 12;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(6, 85);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(176, 13);
            this.label6.TabIndex = 11;
            this.label6.Text = "Automation Script for model Building";
            // 
            // dataset
            // 
            this.dataset.Location = new System.Drawing.Point(188, 56);
            this.dataset.Name = "dataset";
            this.dataset.Size = new System.Drawing.Size(231, 20);
            this.dataset.TabIndex = 8;
            // 
            // pythonLocation
            // 
            this.pythonLocation.Location = new System.Drawing.Point(188, 30);
            this.pythonLocation.Name = "pythonLocation";
            this.pythonLocation.Size = new System.Drawing.Size(231, 20);
            this.pythonLocation.TabIndex = 7;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(11, 33);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(81, 13);
            this.label10.TabIndex = 7;
            this.label10.Text = "Algorithm Used:";
            // 
            // txtAlgUsed
            // 
            this.txtAlgUsed.Location = new System.Drawing.Point(98, 30);
            this.txtAlgUsed.Name = "txtAlgUsed";
            this.txtAlgUsed.Size = new System.Drawing.Size(216, 20);
            this.txtAlgUsed.TabIndex = 8;
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.label14);
            this.groupBox3.Controls.Add(this.txtAccuracy);
            this.groupBox3.Controls.Add(this.txtROC);
            this.groupBox3.Controls.Add(this.txtF1);
            this.groupBox3.Controls.Add(this.txtPrecision);
            this.groupBox3.Controls.Add(this.txtRecall);
            this.groupBox3.Controls.Add(this.label13);
            this.groupBox3.Controls.Add(this.label12);
            this.groupBox3.Controls.Add(this.label11);
            this.groupBox3.Controls.Add(this.label9);
            this.groupBox3.Controls.Add(this.txtAlgUsed);
            this.groupBox3.Controls.Add(this.label10);
            this.groupBox3.Location = new System.Drawing.Point(486, 75);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(320, 165);
            this.groupBox3.TabIndex = 9;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Developed Model Details:";
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(22, 123);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(55, 13);
            this.label14.TabIndex = 18;
            this.label14.Text = "Accuracy:";
            // 
            // txtAccuracy
            // 
            this.txtAccuracy.Location = new System.Drawing.Point(83, 119);
            this.txtAccuracy.Name = "txtAccuracy";
            this.txtAccuracy.Size = new System.Drawing.Size(66, 20);
            this.txtAccuracy.TabIndex = 17;
            // 
            // txtROC
            // 
            this.txtROC.Location = new System.Drawing.Point(233, 93);
            this.txtROC.Name = "txtROC";
            this.txtROC.Size = new System.Drawing.Size(64, 20);
            this.txtROC.TabIndex = 16;
            // 
            // txtF1
            // 
            this.txtF1.Location = new System.Drawing.Point(233, 65);
            this.txtF1.Name = "txtF1";
            this.txtF1.Size = new System.Drawing.Size(64, 20);
            this.txtF1.TabIndex = 15;
            // 
            // txtPrecision
            // 
            this.txtPrecision.Location = new System.Drawing.Point(83, 93);
            this.txtPrecision.Name = "txtPrecision";
            this.txtPrecision.Size = new System.Drawing.Size(66, 20);
            this.txtPrecision.TabIndex = 14;
            // 
            // txtRecall
            // 
            this.txtRecall.Location = new System.Drawing.Point(83, 65);
            this.txtRecall.Name = "txtRecall";
            this.txtRecall.Size = new System.Drawing.Size(66, 20);
            this.txtRecall.TabIndex = 13;
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(169, 96);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(58, 13);
            this.label13.TabIndex = 12;
            this.label13.Text = "ROC AUC:";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(174, 68);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(53, 13);
            this.label12.TabIndex = 11;
            this.label12.Text = "F1-Score:";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(24, 96);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(53, 13);
            this.label11.TabIndex = 10;
            this.label11.Text = "Precision:";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(37, 68);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(40, 13);
            this.label9.TabIndex = 9;
            this.label9.Text = "Recall:";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(822, 350);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.txtStatus);
            this.Controls.Add(this.btnCreateModel);
            this.Name = "Form1";
            this.Text = "Predict Late Delivery  Model ";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnCreateModel;
        private System.Windows.Forms.TextBox txtStatus;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.TextBox dataset;
        private System.Windows.Forms.TextBox pythonLocation;
        private System.Windows.Forms.TextBox script;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox txtTargetFeature;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.TextBox txtVarianceThreshold;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtCoeffMatrix;
        private System.Windows.Forms.TextBox txtCardinality;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox txtAlgUsed;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.TextBox txtAccuracy;
        private System.Windows.Forms.TextBox txtROC;
        private System.Windows.Forms.TextBox txtF1;
        private System.Windows.Forms.TextBox txtPrecision;
        private System.Windows.Forms.TextBox txtRecall;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label9;
    }
}

