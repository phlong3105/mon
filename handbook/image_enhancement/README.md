<div align="center">
<img width="800" src="data/image_enhancement.png">

Image Enhancement
=============================

</div>

- Image enhancement is the procedure of improving the quality and the information 
content of original data before processing.
- In the new era of deep learning, deep image enhancement models can perform a 
variety of tasks such as low-light enhancement, de-rain, de-snow, de-haze, etc.

## Methods

| Status | Method                      | Method Type   | Task                                    | Date       | Publication                     |
|:------:|-----------------------------|---------------|-----------------------------------------|------------|---------------------------------|
|   🟩   | [**Zero-DCE**](zero_dce.md) | Deep Learning | `Low-light`                             | 2020/06/19 | CVPR&nbsp;2020, TPAMI&nbsp;2021 |
|   🟩   | **MRPNet**                  | Deep Learning | `Derain`, `Desnow`, `Dehaze`, `Denoise` | 2021/06/25 | CVPR&nbsp;2021                  |
|   🟩   | [**HINet**](hinet.md)       | Deep Learning | `Derain`, `Deblur`, `Denoise`           | 2021/06/25 | CVPR&nbsp;2021                  |


<table>
	<tr>
        <th rowspan="2">Status</th>
        <th rowspan="2">Method</th>
		<th colspan="2">Architecture</th>
		<th colspan="6">Task</th>
		<th rowspan="2">Date</th>
		<th rowspan="2">Publication</th>
    </tr>
	<tr>
  		<td align="center" nowrap><code>Deep</code></td>
		<td align="center" nowrap><code><nobr>Non-deep</nobr></code></td>
  		<td align="center" nowrap><code>Low-light</code></td>
		<td align="center" nowrap><code>Deblur</code></td>
		<td align="center" nowrap><code>Denoise</code></td>
  		<td align="center" nowrap><code>Derain</code></td>
  		<td align="center" nowrap><code>Desnow</code></td>
  		<td align="center" nowrap><code>Dehaze</code></td>
 	</tr>
	<tr>
  		<td align="center">🟩</td>
		<td nowrap>
			<a href="https://github.com/phlong3105/one/blob/master/handbook/image_enhancement/zero_dce.md"><b>Zero-DCE</b></a>
		</td>
  		<td align="center">x</td>
		<td align="center">&nbsp;</td>
		<td align="center">x</td>
  		<td align="center">&nbsp;</td>
  		<td align="center">&nbsp;</td>
  		<td align="center">&nbsp;</td>
  		<td align="center">&nbsp;</td>    
  		<td align="center">&nbsp;</td>    
  		<td>2020/06/19</td>    
  		<td>CVPR&nbsp;2020, TPAMI&nbsp;2021</td>
 	</tr>
	<tr>
  		<td align="center">🟩</td>
		<td nowrap>
			<a href="https://github.com/phlong3105/one/blob/master/handbook/image_enhancement/mprnet.md"><b>MRPNet</b></a>
		</td>
  		<td align="center">x</td>
		<td align="center">&nbsp;</td>
		<td align="center">&nbsp;</td>
  		<td align="center">&nbsp;</td>
  		<td align="center">x</td>
  		<td align="center">x</td>
  		<td align="center">x</td>    
  		<td align="center">x</td>    
  		<td>2021/06/25</td>    
  		<td>CVPR&nbsp;2021</td>
 	</tr>
	<tr>
  		<td align="center">🟩</td>
		<td nowrap>
			<a href="https://github.com/phlong3105/one/blob/master/handbook/image_enhancement/hinet.md"><b>HINet</b></a>
		</td>
  		<td align="center">x</td>
		<td align="center">&nbsp;</td>
		<td align="center">&nbsp;</td>
  		<td align="center">x</td>
  		<td align="center">x</td>
  		<td align="center">x</td>
  		<td align="center">&nbsp;</td>    
  		<td align="center">&nbsp;</td>    
  		<td>2021/06/25</td>    
  		<td>CVPR&nbsp;2021</td>
 	</tr>
</table>

## Data
