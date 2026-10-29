import unittest
import numpy as np
from scipy import ndimage
import evaluate as E

class EvaluationTests(unittest.TestCase):
    def test_original_svs_coordinates_and_anisotropic_mpp(self):
        pts=np.array([[80,40],[160,40],[160,80],[80,80]])
        mask=E.region_mask((30,30),pts,(.5,1.))
        ys,xs=np.nonzero(mask)
        self.assertEqual((xs.min(),xs.max(),ys.min(),ys.max()),(5,10,5,10))

    def test_frozen_acceptance_matches_existing_detector(self):
        counts={k:np.zeros((96,96),np.float32) for k in ['B','T','Other','Macrophage']}
        counts['Other'][20:50,20:50]=2
        counts['B'][20:50,20:50]=.1
        counts['Other'][60:90,60:90]=2
        inputs={'n':96,'counts':counts,'valid':np.ones((96,96),bool),
                'reference':np.zeros((96,96),bool),'gc':np.zeros((96,96),bool)}
        params=E.PARAMS[E.A.TWO_STAGE]
        original=E.A.detect_two_stage(inputs,**params)[0]
        proposals=E.A.detect(inputs,None,**E.PARAMS['All nuclei (no classifier)'])[0]
        labels,n=ndimage.label(proposals)
        mass=np.bincount(labels.ravel(),weights=sum(counts.values()).ravel(),minlength=n+1)
        bm=np.bincount(labels.ravel(),weights=counts['B'].ravel(),minlength=n+1)
        keep=np.divide(bm,mass,out=np.zeros_like(mass),where=mass>0)>=params['b_share'];keep[0]=False
        np.testing.assert_array_equal(original,keep[labels])
        self.assertTrue(original[35,35]);self.assertFalse(original[75,75])

    def test_resampling_preserves_global_pixel_coordinates(self):
        # Inverse pixel transform of warpAffine must land at x*target/native,
        # including fractional native origin, for both 20x and 40x images.
        for mpp in [.2498,.5005]:
            x=768-128;sx=int(np.floor(x*E.TARGET/mpp))
            shift=sx*mpp/E.TARGET-x
            self.assertAlmostEqual((0-shift)/(mpp/E.TARGET)+sx,x*E.TARGET/mpp)

if __name__=='__main__':unittest.main()
