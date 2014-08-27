# -*- coding: utf8 -*-
#
# Module MESH
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The mesh module provides mesh generators: methods that return a topology and an
accompanying geometry function. Meshes can either be generated on the fly, e.g.
:func:`rectilinear`, or read from external an externally prepared file,
:func:`gmesh`, :func:`igatool`, and converted to nutils format. Note that no
mesh writers are provided at this point; output is handled by the
:mod:`nutils.plot` module.
"""

from . import topology, function, util, element, numpy, numeric, transform, rational, log, _
import os, warnings

# MESH GENERATORS

def rectilinear( richshape, periodic=(), name='rect' ):
  'rectilinear mesh'

  ndims = len(richshape)
  shape = []
  offset = []
  scale = []
  uniform = True
  for v in richshape:
    if isinstance( v, int ):
      assert v > 0
      shape.append( v )
      scale.append( 1 )
      offset.append( 0 )
    elif numpy.equal( v, numpy.linspace(v[0],v[-1],len(v)) ).all():
      shape.append( len(v)-1 )
      scale.append( (v[-1]-v[0]) / float(len(v)-1) )
      offset.append( v[0] )
    else:
      shape.append( len(v)-1 )
      uniform = False
  if all( o == 0 for o in offset[1:] ):
    offset = 0
  if all( s == scale[0] for s in scale[1:] ):
    scale = scale[0]
  indices = numeric.grid( shape )
  structure = numpy.empty( indices.shape[1:], dtype=object )

  if isinstance( name, str ):
    wrap = tuple( sh if i in periodic else 0 for i, sh in enumerate(shape) )
    root = transform.RootTrans( name, wrap )
  else:
    assert all( ( name.take(0,i) == name.take(2,i) ).all() for i in periodic )
    root = transform.RootTransEdges( name, shape )

  reference = element.SimplexReference(1)**ndims
  for index in indices.reshape( ndims, -1 ).T:
    structure[tuple(index)] = element.Element( reference, root << transform.shift(index) )
  topo = topology.StructuredTopology( structure, periodic=periodic )
  if uniform:
    geom = function.ElemFunc( ndims ) * scale + offset
  else:
    funcsp = topo.splinefunc( degree=1, periodic=() )
    coords = numeric.meshgrid( *richshape ).reshape( ndims, -1 )
    geom = ( funcsp * coords ).sum()
  return topo, geom

def revolve( topo, coords, nelems, degree=3, axis=0 ):
  'revolve coordinates'

  # This is a hack. We need to be able to properly multiply topologies.
  DEGREE = (2,) # Degree of coords element

  structure = numpy.array([ [ element.QuadElement( ndims=topo.ndims+1 ) for elem in topo ] for ielem in range(nelems) ])
  revolved_topo = topology.StructuredTopology( structure.reshape( nelems, *topo.structure.shape ), periodic=0 )
  if nelems % 2 == 0:
    revolved_topo.groups[ 'top' ] = revolved_topo[:nelems//2]
    revolved_topo.groups[ 'bottom' ] = revolved_topo[nelems//2:]

  print 'topo:', revolved_topo.structure.shape
  revolved_func = revolved_topo.splinefunc( degree=(degree,)+DEGREE )

  assert isinstance( coords, function.StaticDot )
  assert coords.array.ndim == 2
  nvertices, ndims = coords.array.shape

  phi = ( .5 + numpy.arange(nelems) - .5*degree ) * ( 2 * numpy.pi / nelems )
  weights = numpy.empty(( nelems, nvertices, ndims+1 ))
  weights[...,:axis] = coords.array[:,:axis]
  weights[...,axis] = numpy.cos(phi)[:,_] * coords.array[:,axis]
  weights[...,axis+1] = numpy.sin(phi)[:,_] * coords.array[:,axis]
  weights[...,axis+2:] = coords.array[:,axis+1:]
  weights = numeric.reshape( weights, 2, 1 )

  return revolved_topo, revolved_func.dot( weights )

def gmesh( fname, tags={}, name=None, use_elementary=False ):
  'gmesh'

  if isinstance(fname,str):
    lines = iter(open(fname,'r'))
  else:
    lines = iter(fname)
    fname = 'mesh'

  if name is None:
    name = os.path.basename(fname)

  if isinstance( tags, str ):
    warnings.warn('String format for groups is depricated, please use dictionary format instead with (key,value)=(physical ID,group name)',DeprecationWarning)
    tags = { i+1: tag for i, tag in enumerate( tags.split(',') ) }

  #Parse the file
  sections = {}
  for line in lines:
    line = line.strip()
    assert line[0]=='$'
    sname = line[1:]
    slines = []
    for sline in lines:
      sline = sline.strip()
      if sline=='$End%s'%sname:
        break
      slines.append( sline ) 
    sections[sname] = slines  
        
  #Nodes
  nodedata = sections.pop('Nodes')
  nnodes = int(nodedata.pop(0))
  assert len(nodedata)==nnodes
        
  coords = numpy.empty((nnodes,3))
  coords[:] = numpy.nan
  nidmap = {}
  for line in nodedata:
    words = line.split()
    nid = len(nidmap)
    nidmap[int(words[0])] = nid
    coords[nid] = map( float, words[1:] )
  assert not numpy.isnan(coords).any()
  assert numpy.all( coords[:,2] ) == 0, 'ndims=3 case not yet implemented.'
  coords = coords[:,:2]

  #Elements
  elemdata = sections.pop('Elements')
  nelems = int(elemdata.pop(0))
  assert len(elemdata)==nelems

  elems = {}
  elemgroups = {}
  edges = {}
  edgegroups = {}
  fmap = {}
  nmap = {}
  for line in elemdata:
    words = line.split()
    etype = int(words[1] )
    ntags = int( words[2] )
    if use_elementary:
      assert words[3] == '0', 'option use_elementary=True conflicts with non-zero physical tag'
      tag = tags.get( int( words[4] ), 'elementary' + words[4] )
    else:
      tag = tags.get( int( words[3] ), 'physical' + words[3] )

    nids = numpy.array([ nidmap[int(gmshid)] for gmshid in words[3+ntags:] ])
    elemkey = tuple(sorted(nids))
    if etype == 1: # Linear line
      edgegroups.setdefault(tag,[]).append(elemkey)
    elif etype == 2: # linear triangle
      assert len(nids)==3
      try:
        elem = elems[elemkey]
      except KeyError:
        elemcoords = coords[nids]
        if numpy.linalg.det( elemcoords[:2] - elemcoords[2] ) < 0:
          nids[:2] = nids[1], nids[0]
        ref = element.SimplexReference(2)
        maptrans = transform.MapTrans(ref.vertices,nids)
        elem = element.Element(ref,maptrans)
        elems[elemkey] = elem
      
        fmap[maptrans] = (ref.stdfunc(1),None),
        nmap[maptrans] = nids
        for iedge, edge in enumerate([[1,2],[0,2],[0,1]]):
          edgekey = tuple(sorted(nids[edge]))
          try:
            edges.pop( edgekey )
          except KeyError:
            edges[edgekey] = elem.edge(iedge)
      elemgroups.setdefault(tag,[]).append(elem)
    elif etype == 15:
      assert use_elementary, 'Boundary vertex encountered in mesh with elementary groups.'
    else:
      raise NotImplementedError('Unknown GMSH element type %i' % etype)

  topo = topology.Topology( elems.values() )
  topo.groups = elemgroups
  topo.boundary = topology.Topology( edges.values() )
  topo.boundary.groups = { tag: topology.Topology([ edges[edgekey] for edgekey in edgekeys ]) for tag, edgekeys in edgegroups.items() }

  for tag in tags.values():
    if tag not in topo.groups and tag not in topo.boundary.groups:
      warnings.warn('tag %r defined but not used' % tag )

  log.info('Parsed GMSH file:')
  log.info('Nodes (#%d)' % nnodes)
  log.info('Topology (#%d) with groups: %s' % (len(topo), ', '.join('%s (#%d)' % (name,len(subtopo)) for name,subtopo in topo.groups.items())))
  log.info('Boundary (#%d) with groups: %s' % (len(topo.boundary), ', '.join('%s (#%d)' % (name,len(subtopo)) for name,subtopo in topo.boundary.groups.items())))

  linearfunc = function.function( fmap=fmap, nmap=nmap, ndofs=nnodes, ndims=topo.ndims )
  geom = ( linearfunc[:,_] * coords ).sum(0)
  return topo, geom

def triangulation( vertices, nvertices ):
  'triangulation'

  raise NotImplementedError

  bedges = {}
  nmap = {}
  I = numpy.array( [[2,0],[0,1],[1,2]] )
  for n123 in vertices:
    elem = element.TriangularElement()
    nmap[ elem ] = n123
    for iedge, (n1,n2) in enumerate( n123[I] ):
      try:
        del bedges[ (n2,n1) ]
      except KeyError:
        bedges[ (n1,n2) ] = elem, iedge

  dofaxis = function.DofAxis( nvertices, nmap )
  stdelem = element.PolyTriangle( 1 )
  linearfunc = function.Function( dofaxis=dofaxis, stdmap=dict.fromkeys(nmap,stdelem) )

  connectivity = dict( bedges.iterkeys() )
  N = list( connectivity.popitem() )
  while connectivity:
    N.append( connectivity.pop( N[-1] ) )
  assert N[0] == N[-1]

  structure = []
  for n12 in zip( N[:-1], N[1:] ):
    elem, iedge = bedges[ n12 ]
    structure.append( elem.edge( iedge ) )
    
  topo = topology.Topology( nmap )
  topo.boundary = topology.StructuredTopology( structure, periodic=(1,) )
  return topo

def igatool( path, name=None ):
  'igatool mesh'

  if name is None:
    name = os.path.basename(path)

  import vtk

  reader = vtk.vtkXMLUnstructuredGridReader()
  reader.SetFileName( path )
  reader.Update()

  mesh = reader.GetOutput()

  FieldData = mesh.GetFieldData()
  CellData = mesh.GetCellData()

  NumberOfPoints = int( mesh.GetNumberOfPoints() )
  NumberOfElements = mesh.GetNumberOfCells()
  NumberOfArrays = FieldData.GetNumberOfArrays()

  points = util.arraymap( mesh.GetPoint, float, range(NumberOfPoints) )
  Cij = FieldData.GetArray( 'Cij' )
  Cv = FieldData.GetArray( 'Cv' )
  Cindi = CellData.GetArray( 'Elem_extr_indi')

  elements = []
  degree = 3
  ndims = 2
  nmap = {}
  fmap = {}

  poly = element.PolyLine( element.PolyLine.bernstein_poly( degree ) )**ndims

  for ielem in range(NumberOfElements):

    cellpoints = vtk.vtkIdList()
    mesh.GetCellPoints( ielem, cellpoints )
    nids = util.arraymap( cellpoints.GetId, int, range(cellpoints.GetNumberOfIds()) )

    assert mesh.GetCellType(ielem) == vtk.VTK_HIGHER_ORDER_QUAD
    nb = (degree+1)**2
    assert len(nids) == nb

    n = range( *util.arraymap( Cindi.GetComponent, int, ielem, [0,1] ) )
    I = util.arraymap( Cij.GetComponent, int, n, 0 )
    J = util.arraymap( Cij.GetComponent, int, n, 1 )
    Ce = numpy.zeros(( nb, nb ))
    Ce[I,J] = util.arraymap( Cv.GetComponent, float, n, 0 )

    vertices = [ element.PrimaryVertex( '%s(%d:%d)' % (name,ielem,ivertex) ) for ivertex in range(2**ndims) ]
    elem = element.QuadElement( vertices=vertices, ndims=ndims )
    elements.append( elem )

    fmap[ elem ] = element.ExtractionWrapper( poly, Ce.T )
    nmap[ elem ] = nids

  splinefunc = function.function( fmap, nmap, NumberOfPoints, ndims )

  boundaries = {}
  elemgroups = {}
  vertexgroups = {}
  renumber   = (0,3,1,2)
  for iarray in range( NumberOfArrays ):
    name = FieldData.GetArrayName( iarray )
    index = name.find( '_group_' )
    if index == -1:
      continue
    grouptype = name[:index]
    groupname = name[index+7:]
    A = FieldData.GetArray( iarray )
    I = util.arraymap( A.GetComponent, int, range(A.GetSize()), 0 )
    if grouptype == 'edge':
      belements = [ elements[i//4].edge( renumber[i%4] ) for i in I ]
      boundaries[ groupname ] = topology.Topology( belements )
    elif grouptype == 'vertex':
      vertexgroups[ groupname ] = I
    elif grouptype == 'element':
      elemgroups[ groupname ] = topology.Topology( elements[i] for i in I )
    else:
      raise Exception, 'unknown group type: %r' % grouptype

  topo = topology.Topology( elements )
  topo.groups = elemgroups
  if boundaries:
    topo.boundary = topology.Topology( elem for topo in boundaries.values() for elem in topo )
    topo.boundary.groups = boundaries

  for group in elemgroups.values():
    myboundaries = {}
    for name, boundary in boundaries.iteritems():
      belems = [ belem for belem in boundary.elements if belem.parent[0] in group ]
      if belems:
        myboundaries[ name ] = topology.Topology( belems )
    if myboundaries:
      group.boundary = topology.Topology( elem for topo in myboundaries.values() for elem in topo )
      group.boundary.groups = myboundaries

  funcsp = topo.splinefunc( degree=degree )
  coords = ( funcsp[:,_] * points ).sum( 0 )
  return topo, coords #, vertexgroups

def fromfunc( func, nelems, ndims, degree=1 ):
  'piecewise'

  if isinstance( nelems, int ):
    nelems = [ nelems ]
  assert len( nelems ) == func.func_code.co_argcount
  topo, ref = rectilinear( [ numpy.linspace(0,1,n+1) for n in nelems ] )
  funcsp = topo.splinefunc( degree=degree ).vector( ndims )
  coords = topo.projection( func, onto=funcsp, coords=ref, exact_boundaries=True )
  return topo, coords

def demo( xmin=0, xmax=1, ymin=0, ymax=1 ):
  'demo triangulation of a rectangle'

  phi = numpy.arange( 1.5, 13 ) * (2*numpy.pi) / 12
  P = numpy.array([ numpy.cos(phi), numpy.sin(phi) ])
  P /= abs(P).max(axis=0)
  phi = numpy.arange( 1, 9 ) * (2*numpy.pi) / 8
  Q = numpy.array([ numpy.cos(phi), numpy.sin(phi) ])
  Q /= 2 * numpy.sqrt( abs(Q).max(axis=0) / numpy.sqrt(2) )
  R = numpy.zeros([2,1])

  scale = rational.Scalar([1,1,1])
  coords = numeric.round( numpy.hstack( [P,Q,R] ).T * float(scale) )

  vertices = numpy.array(
    [ ( i, (i+1)%12, 12+(i-i//3)%8 )   for i in range(12) ]
  + [ ( 12+(i+1)%8, 12+i, i+1+(i//2) ) for i in range( 8) ]
  + [ ( 12+i, 12+(i+1)%8, 20 )         for i in range( 8) ] )
  
  elements = []
  root = transform.RootTrans( 'demo', shape=(0,0) )
  reference = element.SimplexReference(2)
  for ielem, elemvertices in enumerate( vertices ):
    elemcoords = coords[ numpy.array(elemvertices) ]
    trans = transform.shift(elemcoords[2],scale) << transform.linear((elemcoords[:2]-elemcoords[2]).T,scale)
    elem = element.Element( reference, root << trans )
    elements.append( elem )

  belems = [ elem.edge(0) for elem in elements[:12] ]
  bgroups = { 'top': belems[0:3], 'left': belems[3:6], 'bottom': belems[6:9], 'right': belems[9:12] }

  topo = topology.Topology( elements )
  topo.boundary = topology.Topology( belems )
  topo.boundary.groups = dict( ( tag, topology.Topology( group ) ) for tag, group in bgroups.items() )

  geom = [.5*(xmin+xmax),.5*(ymin+ymax)] \
       + [.5*(xmax-xmin),.5*(ymax-ymin)] * function.ElemFunc( 2 )

  return topo, geom

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
