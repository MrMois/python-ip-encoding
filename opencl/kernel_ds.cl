

#define SIZE_I  480
#define SIZE_O  96
#define DS_RATE 5


__kernel void downsample(__read_only image2d_t img_i, __write_only image2d_t img_o, int thresh)
{

  const int x_o = get_global_id(0);
  const int y_o = get_global_id(1);

  const int x_tl_i = x_o * DS_RATE;
  const int y_tl_i = y_o * DS_RATE;

  for(int i = 0; i < DS_RATE; i++)
  {
    for(int j = 0; j < DS_RATE; j++)
    {
      
    }
  }




  write_imagef(
    img_o,
    pos_o,

    )

}
