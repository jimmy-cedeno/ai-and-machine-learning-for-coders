package com.example.firsttflite

import android.app.AlertDialog
import android.content.DialogInterface
import android.content.res.AssetManager
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.firsttflite.ui.theme.FirstTFLiteTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.lang.Exception
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {
    private lateinit var tflite : Interpreter
    private lateinit var tflitemodel: ByteBuffer
    private lateinit var txtValue: EditText
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        enableEdgeToEdge()
        try {
            tflitemodel = loadModelFile(this.assets, "model.tflite")
            print("JIMMMY")
            tflite = Interpreter(tflitemodel)
        } catch (ex: Exception) {
            ex.printStackTrace()
        }

        var convertButton: Button = findViewById<Button>(R.id.convertButton)
        convertButton.setOnClickListener {
            doInference()
        }
        txtValue = findViewById<EditText>(R.id.txtValue)
//        setContent {
//            FirstTFLiteTheme {
//                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
//                    Greeting(
//                        name = "World",
//                        modifier = Modifier.padding(innerPadding)
//                    )
//                }
//            }
//        }
    }
    private fun loadModelFile(assetManager: AssetManager, modelPath: String): ByteBuffer{
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun doInference(){
        var userVal: Float = txtValue.text.toString().toFloat()
        var inputVal: FloatArray = floatArrayOf(userVal)
        var outputVal: ByteBuffer = ByteBuffer.allocateDirect(4)
        outputVal.order(ByteOrder.nativeOrder())
        tflite.run(inputVal, outputVal)
        outputVal.rewind()
        var f: Float = outputVal.getFloat()
        val builder = AlertDialog.Builder(this)

        with(builder){
            setTitle("TFLite Interpreter")
            setMessage("Your value is:$f")
            setNeutralButton("OK", DialogInterface.OnClickListener {
                dialog, id -> dialog.cancel()
            })
            show()
        }
    }
}

//@Composable
//fun Greeting(name: String, modifier: Modifier = Modifier) {
//    Text(
//        text = "Hello $name!",
//        modifier = modifier
//    )
//}
//
//@Preview(showBackground = true)
//@Composable
//fun GreetingPreview() {
//    FirstTFLiteTheme {
//        Greeting("Android")
//    }
//}