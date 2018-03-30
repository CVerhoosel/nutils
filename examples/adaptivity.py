#! /usr/bin/env python3

from nutils import *
import fractions, unittest


class MakePlots( object ):

  def __init__( self, geom, exact, optimalrate ):
    self.geom = geom
    self.exact = exact
    self.index = 0
    self.ndofs = []
    self.error_exact = []
    self.error_estimate = []
    self.optimalrate = optimalrate

  def __call__( self, domain, sol, ndofs, error_estimate ):
    self.index += 1

    error_exact = domain['aoi'].integrate( self.exact - sol, geometry=self.geom, ischeme='gauss9' )
    log.user( 'error estimate: %.2e (%.1f%% accurate)' % ( error_estimate, 100.*error_estimate/error_exact ) )

    points, colors = domain.elem_eval( [ self.geom, sol ], ischeme='bezier9', separate=True )
    aoi = domain['aoi'].boundary.elem_eval( self.geom, ischeme='bezier2', separate=True )
    with plot.PyPlot( 'sol', index=self.index ) as plt:
      plt.mesh( points, colors )
      plt.colorbar()
      plt.segments( aoi )

    self.ndofs.append( ndofs )
    self.error_exact.append( error_exact )
    self.error_estimate.append( error_estimate )
    with plot.PyPlot( 'conv', index=self.index ) as plt:
      plt.loglog( self.ndofs, self.error_exact, 'k-^', label='exact' )
      plt.loglog( self.ndofs, self.error_estimate, 'k--', label='estimate' )
      plt.slope_marker( ndofs, min( error_exact, error_estimate ), slope=-self.optimalrate )
      plt.legend( loc=3, frameon=False )
      plt.grid()
      plt.xlabel( 'degrees of freedom' )
      plt.ylabel( 'error' )


def main(
    degree: 'number of elements' = 1,
    circle: 'use circular area of interest (default square)' = False,
    uniform: 'use uniform refinement (default adaptive)' = False,
    basistype: 'basis function' = 'std',
    nrefine: 'maximum allowed number of refinements' = 7,
    figures: 'create figures' = True,
  ):

  # construct domain
  verts = numpy.linspace( -1, 1, 7 )
  basetopo, geom = mesh.rectilinear( [verts,verts] )
  aoi = basetopo.trim( .04 - ((geom+.5)**2).sum(-1), maxrefine=5 ) if circle else basetopo[1:2,1:2]
  domain = ( basetopo.withboundary( outside=... )
           - basetopo[3:,:3].withboundary( inside=... ) ).withsubdomain( aoi=aoi )

  # construct exact sulution (used for boundary conditions and error evaluation)
  exact = ( geom**2 ).sum(-1)**(1./3) * function.sin( (2./3) * function.arctan2(-geom[1],-geom[0]) )
  flux = exact.ngrad( geom )

  # sanity check
  harmonicity = numpy.sqrt( domain.integrate( exact.laplace(geom)**2, geometry=geom, ischeme='gauss9' ) )
  log.info( 'exact solution lsqr harmonicity:', harmonicity )

  # prepare plotting
  makeplots = MakePlots( geom, exact, fractions.Fraction(2 if uniform else degree*3,3) ) if figures else lambda *args, **kwargs: None

  # start adaptive refinement
  for irefine in log.count( 'level', start=1 ):

    # construct, solve course domain primal/dual problem
    basis = domain.basis( basistype, degree=degree )
    laplace = function.outer( basis.grad(geom) ).sum(-1)
    matrix = domain.integrate( laplace, geometry=geom, ischeme='gauss5' )
    rhsprimal = domain.boundary['inside'].integrate( basis * flux, geometry=geom, ischeme='gauss99' )
    rhsdual = domain['aoi'].integrate( basis, geometry=geom, ischeme='gauss5' )
    cons = domain.boundary['outside'].project( exact, ischeme='gauss9', geometry=geom, onto=basis )
    lhsprimal = matrix.solve( rhsprimal, constrain=cons )
    lhsdual = matrix.solve( rhsdual, constrain=cons&0 )
    primal = basis.dot( lhsprimal )
    dual = basis.dot( lhsdual )

    # construct, solve refined domain primal/dual problem
    finedomain = domain.refined
    finebasis = finedomain.basis( basistype, degree=degree )
    finelaplace = function.outer( finebasis.grad(geom) ).sum(-1)
    finematrix = finedomain.integrate( finelaplace, geometry=geom, ischeme='gauss5' )
    finerhsdual = finedomain['aoi'].integrate( finebasis, geometry=geom, ischeme='gauss5' )
    finecons = finedomain.boundary['outside'].project( 0, ischeme='gauss5', geometry=geom, onto=finebasis )
    finelhsdual = finematrix.solve( finerhsdual, constrain=finecons )

    # evaluate error estimate
    dlhsdual = finelhsdual - finedomain.project( dual, onto=finebasis, geometry=geom, ischeme='gauss5' )
    ddualw = finebasis * dlhsdual
    error_est_w = finedomain.boundary['inside'].integrate( ddualw * flux, geometry=geom, ischeme='gauss99' )
    error_est_w -= finedomain.integrate( ( ddualw.grad(geom) * primal.grad(geom) ).sum(-1), geometry=geom, ischeme='gauss5' )

    # plot solution and error convergence
    makeplots( domain, primal, len(lhsprimal), error_estimate=abs(error_est_w).sum() )

    if irefine >= nrefine:
      break

    # refine mesh
    if uniform:
      domain = domain.refined
    else:
      mask = error_est_w**2 > numpy.mean(error_est_w**2)
      domain = domain.refined_by( elem.transform[:-1] for elem in domain.refined.supp( finebasis, mask ) )

  return lhsprimal, error_est_w


class test(unittest.TestCase):

  def test_p1_std(self):
    lhsprimal, error_est_w = main(degree=1, circle=False, uniform=False, basistype='std', nrefine=2, figures=False)
    numeric.assert_allclose64(lhsprimal, 'eNoBhAB7/2s2sDVfNKMjoMtQypXJlzbjNZk0l9Vkyx3Ka'
      'snA0gXL18k0yWjKfsn3yGrKrskdybXIfskeycPIdcj3yLXIdcg6yCQ1PzVdNbE2ZDYDNoE1wDRoM802h'
      'DYoNq019DShM+o2pzZTNuM1PDX2M9bS7MsKN802gTYgNpY1rjR90EbLUcucyv1wQqw=')
    numeric.assert_allclose64(error_est_w, 'eNpjYMAN3BVuyDEwzBVnYOCRZWD4IA0R3XXf4U2S5JE'
      'Hzi855D492yqj9qpckoFhM1ClvAIDQ4QcSM3EZwKK/54Zy9uKBQF1PQGKK4LFzz+/Ir/yxXag2PmHQPO'
      'fgcQSnnJI/5fml/knVi3FwNAAVMcOtC0DLNf+4q5EiVjOi39iH0WBYkD5/UB8Vgy3m78B5epFJ4i1Al2'
      'dATTHRH6ZuLo8SGb1C6C7HjAwGAPxPiBuew4SXfQs/NWFh09f6ChMe2GsYP1y/cNfIvskQTKdjxkY3gD'
      'VTb/PwOAGpLUegUQFnu0VdHssJMyg0CgyT2HaS7dH00SFnoBkdgN1WQJd/x1oWyUQOwNF9wNdEQqWzXm'
      '9Syhcxu2ti/ybFzwKj577y6c8/fzIVfq1nLS0lxRIxR9gqAQCXcwANEMGqP8EMNSigPxicLj/Ed/xaYK'
      '0ikCHrPCbQrk60Q+yT2SWKAk/uqi0/dFhOUUlBgZvYIj+e/LqSabcJplKoD4bGQYGAEjUgPU=')

  def test_p2_spline(self):
    lhsprimal, error_est_w = main(degree=2, circle=False, uniform=False, basistype='spline', nrefine=1, figures=False)
    numeric.assert_allclose64(lhsprimal, 'eNoBbgCR/2s2GTYnNQoz9szZyufJlcl/Ni82OTUuM9jMv'
      '8rSyYHJrzZgNoQ1LDN1zH3Km8lRyek2rDbKNSo0AMwZylrJF8kpN9w2kjbNM4TKrMkKydfIssm2ySHJw'
      'ciVyIfJBsnByHfIV8gXydfIlchXyDrIutg53Q==')
    numeric.assert_allclose64(error_est_w, 'eNpjYMAH5kiUCAXItEvd5rspEywx7dXat0fefeXX+wC'
      'S2yfi/urv83cSi5+1C4i/bn1pKOjFz8SrIQiSq5VpfPL/UepTkYc/5Lc8O/bCSDT75USh02B9qs9mSpx'
      '9qinG8aRDjOu50qvQ147isgKBn0Bya6XPP3V+NOPx3Xu7FB9L1j5eJmb16rxIxBuQHN+TQMmMRwqPWBX'
      'cFOwe+8pESjx9v0WQXwgkxym7SkpYsfnpO9X5D1QVWBUPiMXImLw6J87A4Ke2Q1Hs0UXJTzIr3rCIMjA'
      'ceJL16JbU6ZeKLza9YRJmYPBWeCaTLlcvMu8lv6i4GAMDq+R1SZ7Xpq92iaW/O/mZgaFYVl+yWTLn7RG'
      'xYq4SQcwQAgDpQ2yr')

  def test_p1_std_circle(self):
    lhsprimal, error_est_w = main(degree=1, circle=True, uniform=False, basistype='std', nrefine=1, figures=False)
    numeric.assert_allclose64(lhsprimal, 'eNoBUACv/2s2sDVgNHPpoMtQypXJljbgNZM0RtRkyx3Ka'
      'snMNic25DSm0gnL1sk0yQk3gDaZNcHPYsp+yffIbMq1yR7JtciAyR/JxMh1yPfItch1yDrIoQ4s1w==')
    numeric.assert_allclose64(error_est_w, 'eNpjYMAHJigyMFQrMDBslGBgCJVlYNCVgYhL3mt6FK9'
      'S+njh/VMitbKtzwpl+F9OkYTI/bvFwNB+m4FhoTQDgwVQ7yI5iHj8vXq5dDXW+xPvKj6VVpR4ul5+t/h'
      'kaYjcVmUGhmWqDAwTgepzgXY+kYWIz1SYISmsvOf+OVW9eyWqhx5ckw99ORHohi4VBobG+wwMdx+DVBU'
      '+evxQRuG+XO/zPKB5W4D2KcszMCx+ApZ7rvPE6uXPpw6SLEA/LAPKrwRiBwlMvwIAHKQ+mQ==')


if __name__ == '__main__':
  cli.run(main)
